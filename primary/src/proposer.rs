#![allow(dead_code)]
use std::collections::{BTreeMap, HashMap};

// Copyright(C) Facebook, Inc. and its affiliates.
use crate::messages::{Certificate, ConsensusMessage, Header};
use crate::primary::Height;
use config::{Committee, WorkerId};
use crypto::{Digest, Hash, PublicKey, SignatureService};
use log::{debug, info};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::time::{sleep, Duration, Instant};

#[cfg(test)]
#[path = "tests/proposer_tests.rs"]
pub mod proposer_tests;

/// Soft cap on batches per car. `header_size` is only the early-propose
/// threshold; taking the whole backlog (hundreds of digests) makes a header
/// that cannot be certified, but capping at `header_size` itself starves
/// throughput. Healthy cars carry a handful of batches; 16 is above that
/// and still far below the 200+ cars that froze a lane.
const MAX_DIGESTS_PER_HEADER: usize = 16;

/// The proposer creates new headers and send them to the core for broadcasting and further processing.
pub struct Proposer {
    /// The public key of this primary.
    name: PublicKey,
    /// The committee information
    committee: Committee,
    /// Service to sign headers.
    signature_service: SignatureService,
    /// The size of the headers' payload.
    header_size: usize,
    /// The maximum delay to wait for batches' digests.
    max_header_delay: u64,

    /// Receives the parents to include in the next header (along with their round number).
    rx_core: Receiver<Certificate>,
    /// Receives the batches' digests from our workers.
    rx_workers: Receiver<(Digest, WorkerId, config::BatchMetadata)>,
    // Receives new consensus instance
    rx_instance: Receiver<ConsensusMessage>,
    /// Sends newly created headers to the `Core`.
    tx_core: Sender<Header>,

    /// Receives parameter updates from the `Core`.
    rx_params: Receiver<(usize, u64)>,

    /// The current height of this validator's chain
    height: Height,
    /// Holds the certificate waiting to be included in the next header
    last_parent: Option<Certificate>,
    // Holds the consensus info for the last special header
    consensus_instances: HashMap<Digest, ConsensusMessage>,
    /// Holds the batches' digests waiting to be included in the next header.
    digests: Vec<(Digest, WorkerId, config::BatchMetadata)>,
    /// Keeps track of the size (in bytes) of batches' digests that we received so far.
    payload_size: usize,

    num_active_instances: usize,
    use_special_rule: bool,
    is_special: bool,
}

impl Proposer {
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        name: PublicKey,
        committee: Committee,
        signature_service: SignatureService,
        header_size: usize,
        max_header_delay: u64,
        rx_core: Receiver<Certificate>,
        rx_workers: Receiver<(Digest, WorkerId, config::BatchMetadata)>,
        rx_instance: Receiver<ConsensusMessage>,
        tx_core: Sender<Header>,
        rx_params: Receiver<(usize, u64)>,
    ) {
        /*let genesis: Vec<Digest> = Certificate::genesis(&committee)
        .iter()
        .map(|x| x.digest())
        .collect();*/

        let genesis = Certificate::genesis_cert(&committee);

        tokio::spawn(async move {
            Self {
                name,
                committee,
                signature_service,
                header_size,
                max_header_delay,
                rx_core,
                rx_workers,
                rx_instance,
                tx_core,
                rx_params,
                height: 0,
                last_parent: Some(genesis),
                consensus_instances: HashMap::new(),
                digests: Vec::with_capacity(2 * header_size),
                payload_size: 0,
                num_active_instances: 0,
                use_special_rule: false,
                is_special: false,
            }
            .run()
            .await;
        });
    }

    /// Take the queued digests, but never more than `MAX_DIGESTS_PER_HEADER`.
    fn take_digests_for_header(&mut self) -> Vec<(Digest, WorkerId, config::BatchMetadata)> {
        let take_n = self.digests.len().min(MAX_DIGESTS_PER_HEADER);
        let drained: Vec<_> = self.digests.drain(..take_n).collect();
        let taken_size: usize = drained.iter().map(|(digest, _, _)| digest.size()).sum();
        self.payload_size = self.payload_size.saturating_sub(taken_size);
        drained
    }

    async fn make_header(&mut self) {
        debug!("digests size before is {:?}", self.digests.len());

        let drained = self.take_digests_for_header();
        let payload: BTreeMap<Digest, WorkerId> =
            drained.iter().map(|(d, w, _)| (d.clone(), *w)).collect();
        let batch_metadata: BTreeMap<Digest, config::BatchMetadata> =
            drained.into_iter().map(|(d, _, m)| (d, m)).collect();

        let mut header = Header::new(
            self.name,
            self.height,
            payload,
            batch_metadata,
            self.last_parent.clone().unwrap(),
            &mut self.signature_service,
            self.consensus_instances.clone(),
            self.num_active_instances,
        )
        .await;

        if self.is_special {
            header.special = true;
            //TODO: need to also include the digest of the last proposal. Otherwise there is no gain in latency for that tx.
            // Instead of including Certificate as parent => include digest.
        }

        debug!(
            "make_header: created local header id={:?}, author={:?}, height={}, payload_items={}, consensus_msgs={}",
            header.id,
            header.author,
            header.height,
            header.payload.len(),
            header.consensus_messages.len()
        );

        for (digest, _) in &header.consensus_messages {
            debug!("Header has {:?}", digest);
        }

        #[cfg(feature = "benchmark")]
        for digest in header.payload.keys() {
            // NOTE: This log entry is used to compute performance.
            info!("Created {} -> {:?}", header, digest);
        }

        // Send the new header to the `Core` that will broadcast and process it.
        self.tx_core
            .send(header)
            .await
            .expect("Failed to send header");

        // Reset last parent and consensus state after sending
        self.last_parent = None;
        self.consensus_instances.clear();
        self.num_active_instances = 0;
    }

    // Main loop listening to incoming messages.
    pub async fn run(&mut self) {
        debug!("Dag starting at round {}", self.height);

        let timer = sleep(Duration::from_millis(self.max_header_delay));
        tokio::pin!(timer);
        let mut current_time = Instant::now();

        loop {
            while let Ok((digest, worker_id, metadata)) = self.rx_workers.try_recv() {
                self.payload_size += digest.size();
                self.digests.push((digest, worker_id, metadata));
            }

            // Check if we can propose a new header. We propose a new header when one of the following
            // conditions is met:
            // 1. We have a quorum of certificates from the previous round and enough batches' digests;
            // 2. We have a quorum of certificates from the previous round and the specified maximum
            // inter-header delay has passed.
            // 3. If it is a special block opportunity. That is when either a QC or TC from the previous view forms,
            // we have a ticket to propose a new block
            // For both normal blocks and special blocks, delegate the actual sending to the consensus module
            // in other words core should not be disseminating headers
            //let enough_parents = !self.last_parent.is_empty();
            let enough_parent = self.last_parent.is_some();
            let enough_digests = self.payload_size >= self.header_size;
            let timer_expired = timer.is_elapsed();

            if (timer_expired || enough_digests) && (enough_parent || self.is_special) {
                if timer_expired {
                    debug!("Timer expired for height {}", self.height);
                }

                debug!(
                    "New car proposed after {:?} ms",
                    current_time.elapsed().as_millis()
                );
                debug!("is special is {:?}", self.is_special);
                current_time = Instant::now();

                // Make a new header. Leftover digests keep their payload_size.
                self.make_header().await;

                // Reschedule the timer.
                let deadline = Instant::now() + Duration::from_millis(self.max_header_delay);
                timer.as_mut().reset(deadline);
            }

            tokio::select! {
                // Received info from consensus
                Some(info) = self.rx_instance.recv() => {
                    debug!("received consensus info");

                    match &info {
                        ConsensusMessage::Prepare { slot, view, tc: _, qc_ticket: _, proposals, aggregate_report: _} => {
                            if self.use_special_rule {
                                self.is_special = true;
                            }
                            self.num_active_instances +=1;
                            debug!("prepare has digest: {}", info.digest());
                        },
                        ConsensusMessage::Confirm { slot: _, view: _, qc: _, proposals: _, aggregate_report: _} => {
                            if self.use_special_rule {
                                self.is_special = true;
                            }
                            self.num_active_instances +=1;
                        },
                        _ => {},
                    }

                    self.consensus_instances.insert(info.digest(), info);
                }

                // Receive own certificate from core (we are the author)
                Some(parent) = self.rx_core.recv() => {
                    debug!(
                        "rx_core: received parent certificate at time_now, origin={:?}, height={}, header_digest={:?}",
                        parent.origin(),
                        parent.height,
                        parent.header_digest
                    );

                    if parent.height < self.height {
                        continue;
                    }

                    // Advance to the next height.
                    self.height += 1;
                    debug!("Chain moved to height {}", self.height);

                    // Signal that we have a parent certificates to propose a new header.
                    self.last_parent = Some(parent.clone());
                }

                Some((digest, worker_id, metadata)) = self.rx_workers.recv() => {
                    //println!("   received payload from worker {}", worker_id);
                    self.payload_size += digest.size();
                    self.digests.push((digest, worker_id, metadata));
                }

                // Receive parameter updates from core
                Some((header_size, max_header_delay)) = self.rx_params.recv() => {
                    info!("🔄 Proposer received parameter updates: header_size={}, max_header_delay={}",
                          header_size, max_header_delay);
                    self.header_size = header_size;
                    self.max_header_delay = max_header_delay;
                    info!("✅ Proposer parameters updated successfully");
                }

                () = &mut timer => {
                    // Nothing to do.
                }
            }
        }
    }
}
