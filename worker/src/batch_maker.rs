#![allow(unused_variables)]
#![allow(unused_imports)]
// Copyright(C) Facebook, Inc. and its affiliates.
use crate::quorum_waiter::QuorumWaiterMessage;
use crate::worker::WorkerMessage;
use bytes::Bytes;
//#[cfg(feature = "benchmark")]
use crypto::Digest;
use crypto::PublicKey;
//#[cfg(feature = "benchmark")]
use ed25519_dalek::{Digest as _, Sha512};
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use log::{debug, warn};
//#[cfg(feature = "benchmark")]
use config::get_metrics_logger;
use log::info;
use network::{ReliableSender, SimpleSender};
use primary::{Primary, PrimaryWorkerMessage, WorkerPrimaryMessage};
use std::collections::{HashSet, VecDeque};
//#[cfg(feature = "benchmark")]

use primary::timer::Timer;
use std::convert::TryInto as _;
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use store::Store;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::time::{sleep, Duration, Instant};

#[cfg(test)]
#[path = "tests/batch_maker_tests.rs"]
pub mod batch_maker_tests;

//The message type received by clients
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Transaction {
    pub data: Vec<u8>,
    pub created_at: u128, // timestamp in microseconds since epoch
    pub size: usize,
}
//The message type forwarded to quorum waiters
pub type Batch = Vec<Transaction>;

/// The view number (of consensus)
pub type View = u64;
// The slot (sequence) number of consensus
pub type Slot = u64;

#[derive(Clone, PartialEq, std::fmt::Debug)]
pub enum AsyncEffectType {
    Off = 0,
    TempBlip = 1,  //Send nothing for x seconds, and then release all messages
    Failure = 2,   //Send nothing for x seconds  //TODO: Combine with TempBlip?
    Partition = 3, //Send nothing to partitioned replicas for x seconds, then release all
    Egress = 4,    //For x seconds, delay all outbound messages by some amount
}
fn uint_to_enum(v: u8) -> AsyncEffectType {
    unsafe { std::mem::transmute(v) }
}

/// Assemble clients transactions into batches.
pub struct BatchMaker {
    /// The preferred batch size (in bytes).
    batch_size: usize,
    /// The maximum delay after which to seal the batch (in ms).
    max_batch_delay: u64,
    /// Channel to receive transactions from the network.
    rx_transaction: Receiver<Transaction>,

    //tx_message: Sender<QuorumWaiterMessage>,  /// Output channel to deliver sealed batches to the `QuorumWaiter`.
    tx_batch: Sender<Vec<u8>>, // channel to forward batch digest to processor in order for primary to propose.

    /// The network addresses of the other workers that share our worker id.
    workers_addresses: Vec<(PublicKey, SocketAddr)>,
    /// Holds the current batch.
    current_batch: Batch,
    /// Holds the size of the current batch (in bytes).
    current_batch_size: usize,
    /// A network sender to broadcast the batches to the other workers.
    network: SimpleSender,
    /// Channel to receive parameter updates from worker.
    rx_parameter_updates: Receiver<(usize, u64)>,
    //network: ReliableSender,
    // Currently during asynchrony
    during_simulated_asynchrony: bool,
    // Partition public keys
    partition_public_keys: HashSet<PublicKey>,
    // Partition queue for batch requests
    partition_queue: VecDeque<WorkerMessage>,
    // Store
    store: Store,
    // Quorum waiter
    tx_message: Sender<QuorumWaiterMessage>,
    //Simulating an async event
    pub simulate_asynchrony: bool,
    //Type of effects: 0 for delay full async duration, 1 for partition, 2 for  failure, 3 for egress delay. Will start #type many blips.
    pub asynchrony_type: VecDeque<u8>,
    //Start of async period   //offset from current time (in seconds) when to start next async effect
    pub asynchrony_start: VecDeque<u64>,
    //Duration of async period
    pub asynchrony_duration: VecDeque<u64>,
    ////first k nodes experience specified async behavior
    pub affected_nodes: VecDeque<u64>,
    ////public keys of the other works
    pub keys: Vec<PublicKey>,
    //name of the worker
    pub name: PublicKey,
    // async timer futures
    pub async_timer_futures: FuturesUnordered<Pin<Box<dyn Future<Output = (Slot, View)> + Send>>>,
}

impl BatchMaker {
    pub fn spawn(
        batch_size: usize,
        max_batch_delay: u64,
        rx_transaction: Receiver<Transaction>, //receiver channel from worker.TxReceiverHandler
        tx_message: Sender<QuorumWaiterMessage>, //sender channel to worker.QuorumWaiter
        tx_batch: Sender<Vec<u8>>,             // sender channel to worker.Processor
        workers_addresses: Vec<(PublicKey, SocketAddr)>,
        //partition_public_keys: HashSet<PublicKey>,
        mut store: Store,
        rx_parameter_updates: Receiver<(usize, u64)>, // receiver for parameter updates

        simulate_asynchrony: bool,
        asynchrony_type: VecDeque<u8>,
        asynchrony_start: VecDeque<u64>,
        asynchrony_duration: VecDeque<u64>,
        affected_nodes: VecDeque<u64>,
        keys: Vec<PublicKey>,
        name: PublicKey,
    ) {
        tokio::spawn(async move {
            Self {
                batch_size,
                max_batch_delay,
                rx_transaction,
                tx_message, //previously forwarded batch to Quorum_waiter; now skipping this step.
                tx_batch,
                workers_addresses,
                current_batch: Batch::with_capacity(batch_size * 2),
                current_batch_size: 0,
                network: SimpleSender::new(),
                //network: ReliableSender::new(),
                rx_parameter_updates,
                during_simulated_asynchrony: false,
                //partition_public_keys,
                partition_public_keys: HashSet::new(),
                partition_queue: VecDeque::new(),
                store,
                simulate_asynchrony,
                asynchrony_type,
                asynchrony_start,
                asynchrony_duration,
                affected_nodes,
                keys,
                name,
                async_timer_futures: FuturesUnordered::new(),
            }
            .run()
            .await;
        });
    }

    /// Main loop receiving incoming transactions and creating batches.
    async fn run(&mut self) {
        let timer = sleep(Duration::from_millis(self.max_batch_delay));
        tokio::pin!(timer);

        if self.simulate_asynchrony {
            for i in 0..self.asynchrony_start.len() {
                let start_offset = self.asynchrony_start[i];
                let duration_ms = self.asynchrony_duration[i].saturating_mul(1000);
                let end_offset = start_offset + duration_ms;

                let async_start = Timer::new(0, 0, start_offset);
                let async_end = Timer::new(0, 0, end_offset);

                self.async_timer_futures.push(Box::pin(async_start));
                self.async_timer_futures.push(Box::pin(async_end));

                if uint_to_enum(self.asynchrony_type[i]) == AsyncEffectType::Partition {
                    self.keys.sort();
                    let index = self.keys.binary_search(&self.name).unwrap();

                    // Figure out which partition we are in, partition_nodes indicates when the left partition ends
                    let mut start: usize = 0;
                    let mut end: usize = 0;

                    // We are in the right partition
                    if index > self.affected_nodes[i] as usize - 1 {
                        start = self.affected_nodes[i] as usize;
                        end = self.keys.len();
                    } else {
                        // We are in the left partition
                        start = 0;
                        end = self.affected_nodes[i] as usize;
                    }

                    // These are the nodes in our side of the partition
                    for j in start..end {
                        self.partition_public_keys.insert(self.keys[j]);
                    }

                    debug!("partition pks are {:?}", self.partition_public_keys);
                }
            }
        }

        /*let timer1 = sleep(Duration::from_secs(10));
        tokio::pin!(timer1);
        let timer2 = sleep(Duration::from_secs(30));
        tokio::pin!(timer2);*/
        let mut current_time = Instant::now();

        loop {
            tokio::select! {
                // Assemble client transactions into batches of preset size.
                Some(transaction) = self.rx_transaction.recv() => {
                    self.current_batch_size += transaction.size;
                    self.current_batch.push(transaction);
                    if self.current_batch_size >= self.batch_size {
                        self.seal().await;

                        debug!("batch ready it took {:?} ms", current_time.elapsed().as_millis());
                        current_time = Instant::now();

                        timer.as_mut().reset(Instant::now() + Duration::from_millis(self.max_batch_delay));
                    }
                },

                // If the timer triggers, seal the batch even if it contains few transactions.
                () = &mut timer => {
                    debug!("BatchMaker: max batch delay timer triggered");
                    if !self.current_batch.is_empty() {
                        self.seal().await;
                    }

                    current_time = Instant::now();
                    timer.as_mut().reset(Instant::now() + Duration::from_millis(self.max_batch_delay));
                },

                // Handle parameter updates from worker
                Some((new_batch_size, new_max_batch_delay)) = self.rx_parameter_updates.recv() => {
                    info!("BatchMaker: Updating parameters - batch_size: {} -> {}, max_batch_delay: {} -> {}",
                          self.batch_size, new_batch_size, self.max_batch_delay, new_max_batch_delay);

                    self.batch_size = new_batch_size;
                    self.max_batch_delay = new_max_batch_delay;

                    // Reset timer with new delay
                    timer.as_mut().reset(Instant::now() + Duration::from_millis(self.max_batch_delay));
                },

                Some((slot, view)) = self.async_timer_futures.next() => {
                    self.during_simulated_asynchrony = !self.during_simulated_asynchrony;
                    debug!("partition queue size is {:?}", self.partition_queue.len());
                },

                // If the timer triggers, seal the batch even if it contains few transactions.
                /*() = &mut timer1 => {
                    debug!("BatchMaker: partition delay timer 1 triggered");
                    self.during_simulated_asynchrony = true;
                    timer1.as_mut().reset(Instant::now() + Duration::from_secs(100));
                },

                // If the timer triggers, seal the batch even if it contains few transactions.
                () = &mut timer2 => {
                    debug!("BatchMaker: partition delay timer 2 triggered");
                    debug!("partition queue size is {:?}", self.partition_queue.len());
                    self.during_simulated_asynchrony = false;
                    timer2.as_mut().reset(Instant::now() + Duration::from_secs(100));
                },*/



            }

            // Give the change to schedule other tasks.
            tokio::task::yield_now().await;
        }
    }

    /// Seal and broadcast the current batch.
    async fn seal(&mut self) {
        #[cfg(feature = "benchmark")]
        let size = self.current_batch_size;

        // Get current timestamp for batch creation
        let batch_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        // Look for sample txs (they all start with 0) and gather their txs id (the next 8 bytes).
        #[cfg(feature = "benchmark")]
        let sample_tx_ids: Vec<u64> = self
            .current_batch
            .iter()
            .filter(|tx| tx.data.len() > 0 && tx.data[0] == 0u8 && tx.data.len() > 8)
            .filter_map(|tx| tx.data[1..9].try_into().ok())
            .map(|bytes| u64::from_be_bytes(bytes))
            .collect();

        // Serialize the batch.
        self.current_batch_size = 0;
        let batch: Vec<_> = self.current_batch.drain(..).collect();
        let message = WorkerMessage::Batch(batch.clone());
        let serialized = bincode::serialize(&message).expect("Failed to serialize our own batch");

        #[cfg(feature = "benchmark")]
        {
            // NOTE: This is one extra hash that is only needed to print the following log entries.
            let digest = Digest(
                Sha512::digest(&serialized).as_slice()[..32]
                    .try_into()
                    .unwrap(),
            );

            // Log batch creation with detailed information
            // info!(
            //     "BatchMaker: Created batch {:?} with {} transactions, size={}B, timestamp={}μs",
            //     digest,
            //     batch.len(),
            //     size,
            //     batch_timestamp
            // );

            // // Log individual transactions in the batch
            // for (tx_idx, tx) in batch.iter().enumerate() {
            //     let tx_age_us = batch_timestamp.saturating_sub(tx.created_at.try_into().unwrap());
            //     let tx_age_ms = tx_age_us as f64 / 1000.0;

            //     info!(
            //         "  Batch[{:?}] Tx[{}]: size={}B, created_at={}μs, batch_age={:.2}ms",
            //         digest, tx_idx, tx.size, tx.created_at, tx_age_ms
            //     );
            // }

            // Log sample transaction IDs (for performance analysis)
            for &id in &sample_tx_ids {
                // NOTE: This log entry is used to compute performance.
                info!("Batch {:?} contains sample tx {}", digest, id);
            }

            // NOTE: This log entry is used to compute performance.
            info!("Batch {:?} contains {} B", digest, size);

            // Calculate and log batch statistics
            let total_tx_size: usize = batch.iter().map(|tx| tx.size).sum();
            let avg_tx_size = if !batch.is_empty() {
                total_tx_size as f64 / batch.len() as f64
            } else {
                0.0
            };
            let avg_tx_age: f64 = batch
                .iter()
                .map(|tx| {
                    batch_timestamp.saturating_sub(tx.created_at.try_into().unwrap()) as f64
                        / 1000.0
                })
                .sum::<f64>()
                / batch.len() as f64;

            // info!("Batch[{:?}] stats: total_tx_size={}B, avg_tx_size={:.1}B, avg_tx_age={:.2}ms, sample_txs={}",
            //       digest, total_tx_size, avg_tx_size, avg_tx_age, sample_tx_ids.len());

            // Log batch creation event to metrics.log
            if let Some(mut logger_guard) = get_metrics_logger() {
                if let Some(logger) = logger_guard.as_mut() {
                    let batch_metrics = serde_json::json!({
                        "batch_digest": digest.to_string(),
                        "batch_size_bytes": size,
                        "transaction_count": batch.len(),
                        "total_tx_size_bytes": total_tx_size,
                        "avg_tx_size_bytes": avg_tx_size,
                        "avg_tx_age_ms": avg_tx_age,
                        "sample_tx_count": sample_tx_ids.len(),
                        "sample_tx_ids": sample_tx_ids,
                        "batch_timestamp_us": batch_timestamp,
                        "worker_name": hex::encode(self.name.0)
                    });
                    logger.log_event("batch_created", batch_metrics);
                }
            }
        }

        // Broadcast the batch through the network.

        //NEW:
        //Best-effort broadcast only. Any failure is correlated with the primary operating this node (running on same machine)

        let bytes = Bytes::from(serialized.clone());
        let digest = Digest(
            Sha512::digest(&serialized).as_slice()[..32]
                .try_into()
                .unwrap(),
        );

        // Store the batch.
        self.store.write(digest.to_vec(), serialized.clone()).await;
        if self.tx_batch.try_send(serialized.clone()).is_err() {
            warn!(
                "BatchMaker: processor channel full, dropping batch {:?}",
                digest
            );
        }
        if self.during_simulated_asynchrony {
            debug!("BatchMaker: Simulated asynchrony enabled. Only sending to partitioned keys from broadcast");
            let new_addresses: Vec<_> = self
                .workers_addresses
                .iter()
                .filter(|(pk, _)| self.partition_public_keys.contains(pk))
                .map(|(_, addr)| addr)
                .cloned()
                .collect();
            //let (_, addresses) = new_addresses.iter().cloned().unzip();
            //debug!("addresses is {:?}", new_addresses);
            self.partition_queue.push_back(message);
            debug!("partition queue size is {:?}", self.partition_queue.len());
            self.network.broadcast(new_addresses, bytes).await;
        } else {
            //debug!("sending batch normally");
            let (_, addresses): (Vec<_>, _) = self.workers_addresses.iter().cloned().unzip();
            self.network.broadcast(addresses, bytes).await;
        }

        /*let digest = Digest(Sha512::digest(&serialized).as_slice()[..32].try_into().unwrap());
        self.store.write(digest.to_vec(), serialized.clone()).await;
        self.tx_batch.send(serialized.clone()).await.expect("Failed to deliver batch");

        //OLD:
        //This uses reliable sender. The receiver worker will reply with an ack. The Reply Handler is passed to Quorum Waiter.
        let (names, addresses): (Vec<_>, Vec<_>) = self.workers_addresses.iter().cloned().unzip();
        let new_addresses: Vec<_> = self.workers_addresses.iter().filter(|(pk, _)| self.partition_public_keys.contains(pk)).map(|(_, addr)| addr).cloned().collect();
        let bytes = Bytes::from(serialized.clone());
        //let handlers = self.network.broadcast(addresses, bytes).await;
        let handlers = self.network.broadcast(new_addresses, bytes).await;

        // // Send the batch through the deliver channel for further processing.
        self.tx_message
             .send(QuorumWaiterMessage {
                batch: serialized,
                 handlers: names.into_iter().zip(handlers.into_iter()).collect(),
             })
             .await
            .expect("Failed to deliver batch");*/
    }
}
