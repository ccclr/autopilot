#![allow(dead_code)]
#![allow(unused_variables)]
// Copyright(C) Facebook, Inc. and its affiliates.
use crate::aggregators::{QCMaker, TCMaker, VotesAggregator};
//use crate::common::special_header;
use crate::error::{DagError, DagResult};
use crate::leader::LeaderElector;
use crate::messages::{
    transform_commitQC, verify_commit, verify_confirm, AggregateReport, Certificate, CommitQC,
    ConsensusMessage, ConsensusRequest, ConsensusType, ConsensusVote, Header, Proposal,
    StateReport, Timeout, Vote, QC, TC,
};
use crate::primary::{Height, PrimaryMessage, Slot, View};
use crate::synchronizer::{self, Synchronizer};
use crate::timer::{CarTimer, FastTimer, PayloadTimer, Timer};
use crate::PrimaryWorkerMessage;
use async_recursion::async_recursion;
use bytes::Bytes;
use config::{
    get_metrics_logger, AggregationStrategy, Committee, DataPollutionStrategy, Stake,
    PARAMETER_UPDATE_SIGNAL_FILE,
};
use core::panic;
use crypto::{Digest, PublicKey, SignatureService};
use crypto::{Hash as _, Signature};
use futures::stream::FuturesUnordered;
use futures::{Future, StreamExt};
use log::{debug, error, info, warn};
use network::{CancelHandler, ReliableSender};
use rand::Rng as _;
use serde::{Deserialize, Serialize};
use std::borrow::BorrowMut;
use tokio_util::time::DelayQueue;
//use tokio::time::error::Elapsed;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
//use std::time::{Duration, Instant};
//use std::task::Poll;
use std::cmp::max;
use store::Store;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::process::Command;
use tokio::sync::mpsc::{Receiver, Sender, UnboundedReceiver, UnboundedSender};
use tokio::time::{sleep, Duration, Instant};
//use crate::messages_consensus::{QC, TC};
#[cfg(test)]
#[path = "tests/core_tests.rs"]
pub mod core_tests;

#[derive(Clone, Copy, PartialEq, std::fmt::Debug)]
pub enum AsyncEffectType {
    Off = 0,
    TempBlip = 1,     //Send nothing for x seconds, and then release all messages
    Failure = 2,      //Send nothing for x seconds  //TODO: Combine with TempBlip?
    Partition = 3,    //Send nothing to partitioned replicas for x seconds, then release all
    Egress = 4,       //For x seconds, delay all outbound messages by some amount
    PrepareDelay = 5, //For x seconds, delay all Prepare messages by some amount
    VoteDelay = 6,    //For x seconds, delay all Vote messages by some amount
    Equivocate = 7,   //Send different headers to different targets
}

impl From<u8> for AsyncEffectType {
    fn from(v: u8) -> Self {
        match v {
            0 => AsyncEffectType::Off,
            1 => AsyncEffectType::TempBlip,
            2 => AsyncEffectType::Failure,
            3 => AsyncEffectType::Partition,
            4 => AsyncEffectType::Egress,
            5 => AsyncEffectType::PrepareDelay,
            6 => AsyncEffectType::VoteDelay,
            7 => AsyncEffectType::Equivocate,
            _ => AsyncEffectType::Off,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum CoordinationPath {
    Fast,
    Slow,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CoordinationCertificate {
    epoch: u64,
    slot: Slot,
    view: View,
    aggregate_report_digest: Digest,
    path: CoordinationPath,
    qc_votes: usize,
}

#[derive(Debug, Clone)]
pub struct DelayedMessage {
    pub message: PrimaryMessage,
    pub height: u64,
    pub target: Option<PublicKey>,
    pub is_consensus: bool,
    pub scheduled_time: Instant,
}

pub struct Core {
    /// The public key of this primary.
    name: PublicKey,
    /// The committee information.
    committee: Committee,
    /// The node index of this primary.
    node_index: u64,
    /// The persistent storage.
    store: Store,
    /// Handles synchronization with other nodes and our workers.
    synchronizer: Synchronizer,
    /// Service to sign headers.
    signature_service: SignatureService,
    /// The current consensus round (used for cleanup).
    consensus_round: Arc<AtomicU64>,
    /// The depth of the garbage collector.
    gc_depth: Height,

    /// Receiver for dag messages (headers, votes, certificates).
    rx_primaries: Receiver<PrimaryMessage>,
    /// Receives loopback headers from the `HeaderWaiter`.
    rx_header_waiter: Receiver<Header>,
    /// Receives loopback instances from the 'HeaderWaiter'
    rx_header_waiter_instances: Receiver<(ConsensusMessage, Header)>,
    /// Receives our newly created headers from the `Proposer`.
    rx_proposer: Receiver<Header>,
    // Output all certificates to the consensus Dag view
    tx_committer: Sender<(ConsensusMessage, bool)>,

    /// Send a valid parent certificate to the `Proposer`
    tx_proposer: Sender<Certificate>,
    // Receive sync requests for headers required at the consensus layer
    rx_request_header_sync: Receiver<Digest>,

    /// The last garbage collected round.
    gc_round: Height,

    /// The authors of the last voted headers. (Ensures only voting for one header per round)
    last_voted: HashMap<Height, HashSet<PublicKey>>,
    /// The last header we proposed (for which we are waiting votes).
    current_header: Header,
    // Whether we have already sent certificate to proposer
    sent_cert_to_proposer: bool,

    // /// Aggregates votes into a certificate.
    votes_aggregator: VotesAggregator,

    network: ReliableSender,
    /// Keeps the cancel handlers of the messages we sent.
    cancel_handlers: HashMap<Height, Vec<CancelHandler>>,
    consensus_cancel_handlers: HashMap<Slot, Vec<CancelHandler>>,

    current_proposal_tips: HashMap<PublicKey, Proposal>,
    current_certified_tips: HashMap<PublicKey, Proposal>,

    consensus_instances: HashMap<(Slot, Digest), ConsensusMessage>,
    views: HashMap<Slot, View>,
    timers: HashSet<(Slot, View)>,
    last_voted_consensus: HashSet<(Slot, View)>,
    timer_futures: FuturesUnordered<Pin<Box<dyn Future<Output = (Slot, View)> + Send>>>,
    // TODO: Add garbage collection, related to how deep pipeline (parameter k)
    high_proposals: HashMap<Slot, ConsensusMessage>,
    high_qcs: HashMap<Slot, ConsensusMessage>, // NOTE: Store the latest QC for each slot
    qc_makers: HashMap<(Slot, Digest), QCMaker>,
    // pqc_makers: HashMap<(Slot, View), QCMaker>,
    // cqc_makers: HashMap<(Slot, View), QCMaker>,
    current_qcs_formed: usize,
    tc_makers: HashMap<(Slot, View), TCMaker>,
    prepare_tickets: VecDeque<ConsensusMessage>,
    already_proposed_slots: HashSet<Slot>,
    tx_info: Sender<ConsensusMessage>,
    leader_elector: LeaderElector,
    timeout_delay: u64,
    // GC the vote aggregators and current headers
    // gc_map: HashMap<Round, Digest>,
    committed_slots: HashMap<Slot, CommitQC>,
    last_committed_slot: u64,
    //TODO: if we are not enforcing a ticket, then only start when we committed all instances < s-k.
    // If we just check that s-k is committed, but all it's predecessors are not, then we may still open an arbitrary number of instances in the absolute worst case
    // E.g. s-1 has not committed, but s has, so we can open s+k

    //Configuration options: //TODO: Move to Primary level -> make configurable from main.rs
    use_fast_path: bool,          //default = false
    use_optimistic_tips: bool,    //default = true (TODO: implement non optimistic tip option)
    use_parallel_proposals: bool, //default = true (TODO: implement sequential slot option)
    k: u64, //limit k on number of open honest instances (k+f instances can be open) => if require QC, then hard limit to k.
    last_used_k: u64, // Previous k value used before latest in-epoch parameter application.
    // When k is reduced, keep old k for in-flight slots up to this boundary.
    k_transition_end_slot: Option<u64>,
    // Safer k-decrease: target k while draining in-flight instances opened under the old k.
    // While set, new slots open with pending_k; self.k stays at the old value until activation.
    pending_k: Option<u64>,
    fast_path_timeout: u64,

    use_ride_share: bool,
    car_timeout: u64,
    car_timer_futures: FuturesUnordered<Pin<Box<dyn Future<Output = Vote> + Send>>>,
    fast_timer_futures: FuturesUnordered<Pin<Box<dyn Future<Output = ConsensusVote> + Send>>>, // Use this one for Fast Path on external Consensus case

    cut_condition_type: u8,

    //simulate_asynchrony: bool, //Simulating an async event

    //asynchrony_start: u64,     //Start of async period   //offset from current time (in seconds) when to start next async effect
    //asynchrony_duration: u64,  //Duration of async period
    // during_simulated_asynchrony: bool,  //Currently in async period?
    // async_timer_futures: FuturesUnordered<Pin<Box<dyn Future<Output = (Slot, View)> + Send>>>, //Used to turn on/off period  //Note: (slot, view) are not needed, it's just to re-use existing Timer

    // current_time: Instant,
    // //For full delay
    // async_delayed_prepare: Option<ConsensusMessage>,

    //TODO: Replace with the generic framework.
    //parition simulation

    // partition_public_keys: HashSet<PublicKey>,
    // partition_delayed_msgs: Vec<(PrimaryMessage, u64, Option<PublicKey>, bool)>, //(msg, height, author, consensus/car path)

    // //failure simulation
    // simulate_failure: bool,
    // failure_start: u64,
    // failure_duration: u64,
    // failure_nodes: u64, //first k nodes to fail
    // during_simulated_failure: bool,
    // failure_timer_futures: FuturesUnordered<Pin<Box<dyn Future<Output = (Slot, View)> + Send>>>,
    // //drop messages

    // //egress delay simulation
    // simulate_egress_delay: bool,
    // delay_start: u64,
    // delay_duration: u64,
    // egress_penalty: u64, //the number of ms of egress penalty.
    // delayed_nodes: u64, //first k nodes experience penalty
    // during_simulated_delay: bool,
    // delay_timer_futures: FuturesUnordered<Pin<Box<dyn Future<Output = (Slot, View)> + Send>>>, //Use these timers to turn on/off async period
    // delayed_messages: VecDeque<(u64, PrimaryMessage, u64, Option<PublicKey>, bool)>, //(wake-time, msg, height, author, consensus/car path)
    // egress_timer_futures: FuturesUnordered<Pin<Box<dyn Future<Output = (Slot, View)> + Send>>>, //Use this timer to wake next delayed message.
    //                                                                                             //Use Instant::now().elapsed().as_milis() to get current time to compute wake-time

    // //Asynchrony simulation framework:
    simulate_asynchrony: bool,                  //Simulating an async event
    asynchrony_type: VecDeque<AsyncEffectType>, //Type of effects: 0 for delay full async duration, 1 for partition, 2 for  failure, 3 for egress delay. Will start #type many blips.
    asynchrony_start: VecDeque<u64>, //Start of async period   //offset from current time (in seconds) when to start next async effect
    asynchrony_duration: VecDeque<u64>, //Duration of async period (seconds)
    affected_nodes: VecDeque<u64>,   ////first k nodes experience specified async behavior.
    asynchrony_node_ids_per_window: VecDeque<Vec<u64>>, // Optional explicit node ids for each async window.
    during_simulated_asynchrony: bool,                  //Currently in async period?
    current_effect_type: AsyncEffectType,               //Currently active effect.
    current_asynchrony_node_ids: Vec<u64>, // Explicit node ids for currently active async window.
    current_num_affected_nodes: u64,

    async_timer_futures: FuturesUnordered<Pin<Box<dyn Future<Output = (Slot, View)> + Send>>>, //Used to turn on/off period  //Note: (slot, view) are not needed, it's just to re-use existing Timer
    already_set_timers: bool,

    current_time: Instant,
    //For full delay
    async_delayed_prepare: Option<ConsensusMessage>,
    //For partition
    partition_public_keys: HashSet<PublicKey>,
    partition_delayed_msgs: Vec<(PrimaryMessage, u64, Option<PublicKey>, bool)>, //(msg, height, author, consensus/car path)
    //For egress
    egress_penalty: u64, //the number of ms of egress penalty.
    //delayed_messages: VecDeque<(u128, PrimaryMessage, u64, Option<PublicKey>, bool)>, //(wake-time, msg, height, author, consensus/car path)
    //egress_timer_futures: FuturesUnordered<Pin<Box<dyn Future<Output = (Slot, View)> + Send>>>, //Use this timer to wake next delayed message.
    //                                                                                             //Use Instant::now().elapsed().as_milis() to get current time to compute wake-time
    //egress_timer: Timer,
    //egress_delayed_msgs: VecDeque<(PrimaryMessage, u64, Option<PublicKey>, bool)>,
    egress_delay_queue: DelayQueue<(PrimaryMessage, u64, Option<PublicKey>, bool)>,
    current_egress_end: Instant,
    // exponential timeouts
    use_expoential_timeouts: bool,
    dropped_slot: u64,
    // Channel to communicate async period to the worker
    tx_worker_async_channel: Sender<(bool, HashSet<PublicKey>)>,
    // Batch payload timers
    payload_timer_futures: FuturesUnordered<Pin<Box<dyn Future<Output = Header> + Send>>>,
    // Missed payloads
    missed_payloads: u64,
    target_ip_addresses: VecDeque<String>,
    // Store last few headers for equivocation simulation
    last_headers: VecDeque<Header>,

    // Metrics collection parameters
    epoch_slots: u64, // Number of slots per epoch (h parameter) - DEPRECATED, kept for compatibility
    window_size: u64, // Size of time window within each epoch (j parameter) - DEPRECATED
    last_triggered_slot: u64, // Last slot that triggered metrics collection (to avoid duplicate triggers) - DEPRECATED

    // Transaction-based metrics collection parameters (NEW)
    total_committed_transactions: u64, // Cumulative count of committed transactions
    last_triggered_tx_count: u64,      // Last tx count that triggered metrics collection

    // Slot mapping for current epoch window
    epoch_slot_start: Option<u64>, // Slot when we started recording after last collect
    epoch_slot_end: Option<u64>,   // Slot when we reach window_transactions threshold

    // Channel to receive metrics state from metrics_collector
    rx_metrics_state: Receiver<String>, // Receives JSON state string from metrics_collector
    tx_metrics_state: Sender<String>,   // Sends JSON state string to main loop

    // Persistent connection to metrics_collector
    metrics_collector_sender: Option<tokio::sync::mpsc::UnboundedSender<String>>, // Send requests to metrics_collector without waiting
    metrics_collector_task: Option<tokio::task::JoinHandle<()>>, // Task to listen for metrics_collector connections

    // Collected state reports per epoch
    state_reports: HashMap<u64, HashMap<PublicKey, StateReport>>,

    // Collection timer (async fallback)
    collection_timer_futures: FuturesUnordered<Pin<Box<dyn Future<Output = u64> + Send>>>,
    collection_timeout_ms: u64,
    active_collection_timers: HashSet<u64>,

    // Pending aggregate reports ready to embed in Prepare (epoch -> report).
    // A report remains pending until its CoordinationCertificate is persisted.
    pending_aggregate_reports: BTreeMap<u64, AggregateReport>,
    // Aggregate report bound to consensus instance (slot, view) once verified.
    certifiable_aggregate_reports: HashMap<(Slot, View), AggregateReport>,
    // Epochs with persisted Coordination Certificate.
    certified_aggregate_epochs: HashSet<u64>,
    aggregated_epochs: HashSet<u64>,

    // Unix socket listener for RL parameter update signals
    rl_param_update_sender: UnboundedSender<String>,
    rl_param_update_receiver: UnboundedReceiver<String>,
    rl_param_update_task: Option<tokio::task::JoinHandle<()>>,

    // Request ID counter for matching requests/responses
    next_request_id: u64,

    // RL parameter update signal counter (epoch-indexed)
    rl_param_signal_epoch: u64,

    // Slot in each epoch after which parameter updates are applied
    applied_begin: u64,

    // Pending parameter update epoch (apply when applied_begin slot commits)
    pending_param_update_epoch: Option<u64>,
    // Pending parameter updates captured at signal time
    pending_param_update_params: Option<serde_json::Value>,
    // Whether RL params were installed at applied_begin for param_apply_ok_epoch.
    // StateReport for epoch t piggybacks the bit for epoch t-1: metrics are
    // collected at window_size, which is before this epoch's applied_begin, so
    // a complete (action, reward) pair is "applied in t-1, measured in t".
    param_apply_ok: bool,
    param_apply_ok_epoch: u64,

    // Channel to send parameter updates to proposer
    tx_proposer_params: Sender<(usize, u64)>, // (header_size, max_header_delay)

    // Parameters tracked for dynamic updates (used by proposer and workers)
    header_size: usize,
    max_header_delay: u64,
    batch_size: usize,
    max_batch_delay: u64,

    // Ablation: configurable growth_rate aggregation strategy.
    aggregation_strategy: AggregationStrategy,
    // Ablation: data-pollution simulation.
    data_pollution_node_ids: Vec<u64>,
    data_pollution_prob: f64,
    data_pollution_strategy: DataPollutionStrategy,
}

impl Core {
    #[allow(clippy::too_many_arguments)]
    pub fn spawn(
        name: PublicKey,
        committee: Committee,
        store: Store,
        store_path: String,
        synchronizer: Synchronizer,
        signature_service: SignatureService,
        consensus_round: Arc<AtomicU64>,
        gc_depth: Height,
        rx_primaries: Receiver<PrimaryMessage>,
        rx_header_waiter: Receiver<Header>,
        rx_header_waiter_instances: Receiver<(ConsensusMessage, Header)>,
        rx_proposer: Receiver<Header>,
        tx_committer: Sender<(ConsensusMessage, bool)>,
        tx_proposer: Sender<Certificate>,
        rx_request_header_sync: Receiver<Digest>,
        tx_info: Sender<ConsensusMessage>,
        leader_elector: LeaderElector,
        timeout_delay: u64,
        use_optimistic_tips: bool,
        use_parallel_proposals: bool,
        k: u64,
        use_fast_path: bool,
        fast_path_timeout: u64,
        use_ride_share: bool,
        car_timeout: u64,
        cut_condition_type: u8,
        simulate_asynchrony: bool,
        //asynchrony_start: u64,
        //asynchrony_duration: u64,
        // Temp comment out
        async_type: VecDeque<u8>,
        asynchrony_start: VecDeque<u64>,
        asynchrony_duration: VecDeque<u64>,
        affected_nodes: VecDeque<u64>,
        asynchrony_node_ids_per_window: VecDeque<Vec<u64>>,
        egress_penalty: u64,
        egress_penalty_per_node: VecDeque<u64>,
        use_expoential_timeouts: bool,
        tx_worker_async_channel: Sender<(bool, HashSet<PublicKey>)>,
        target_ip_addresses: VecDeque<String>,
        epoch_slots: u64,
        window_size: u64,
        rx_metrics_state: Receiver<String>,
        tx_metrics_state: Sender<String>,
        tx_proposer_params: Sender<(usize, u64)>,
        header_size: usize,
        max_header_delay: u64,
        batch_size: usize,
        max_batch_delay: u64,
        applied_begin: u64,
        aggregation_strategy: AggregationStrategy,
        data_pollution_node_ids: Vec<u64>,
        data_pollution_prob: f64,
        data_pollution_strategy: DataPollutionStrategy,
    ) {
        tokio::spawn(async move {
            // Extract node index from store path (e.g., ".db-0" -> 0)
            let node_index = store_path
                .rsplit('-')
                .next()
                .and_then(|s| s.parse::<u64>().ok())
                .expect("Failed to extract node index from store path");
            let resolved_egress_penalty = egress_penalty_per_node
                .get(node_index as usize)
                .copied()
                .unwrap_or(egress_penalty);

            let (rl_param_update_sender, rl_param_update_receiver) =
                tokio::sync::mpsc::unbounded_channel::<String>();

            Self {
                name,
                //current_header: Header::genesis(&committee),
                committee,
                node_index,
                store,
                synchronizer,
                signature_service,
                consensus_round,
                gc_depth,
                rx_primaries,
                rx_header_waiter,
                rx_header_waiter_instances,
                rx_proposer,
                tx_committer,
                tx_proposer,
                rx_request_header_sync,
                tx_info,
                leader_elector,
                gc_round: 0,
                current_qcs_formed: 0,
                sent_cert_to_proposer: false,
                last_voted: HashMap::with_capacity(2 * gc_depth as usize),
                current_header: Header::default(),
                votes_aggregator: VotesAggregator::new(),
                network: ReliableSender::new(),
                cancel_handlers: HashMap::with_capacity(2 * gc_depth as usize),
                consensus_cancel_handlers: HashMap::with_capacity(2 * gc_depth as usize),
                already_proposed_slots: HashSet::new(),
                current_proposal_tips: HashMap::with_capacity(2 * gc_depth as usize),
                current_certified_tips: HashMap::with_capacity(2 * gc_depth as usize),
                consensus_instances: HashMap::with_capacity(2 * gc_depth as usize),
                views: HashMap::with_capacity(2 * gc_depth as usize),
                timers: HashSet::with_capacity(2 * gc_depth as usize),
                last_voted_consensus: HashSet::with_capacity(2 * gc_depth as usize),
                high_qcs: HashMap::with_capacity(2 * gc_depth as usize),
                high_proposals: HashMap::with_capacity(2 * gc_depth as usize),
                qc_makers: HashMap::with_capacity(2 * gc_depth as usize),
                // pqc_makers: HashMap::with_capacity(2 * gc_depth as usize),
                // cqc_makers: HashMap::with_capacity(2 * gc_depth as usize),
                tc_makers: HashMap::with_capacity(2 * gc_depth as usize),
                prepare_tickets: VecDeque::with_capacity(2 * gc_depth as usize),
                timeout_delay,
                timer_futures: FuturesUnordered::new(),
                //gc_map: HashMap::with_capacity(2 * gc_depth as usize),
                committed_slots: HashMap::with_capacity(2 * gc_depth as usize),
                last_committed_slot: 0,

                use_fast_path,          //default = true
                use_optimistic_tips,    //default = true (TODO: implement non optimistic tip option)
                use_parallel_proposals, //default = true (TODO: implement sequential slot option)
                k,
                last_used_k: k,
                k_transition_end_slot: None,
                pending_k: None,
                fast_path_timeout,
                use_ride_share,
                car_timeout,
                car_timer_futures: FuturesUnordered::new(),
                fast_timer_futures: FuturesUnordered::new(),
                already_set_timers: false,
                cut_condition_type,
                //simulate_asynchrony,
                // asynchrony_start,
                // asynchrony_duration,
                //during_simulated_asynchrony: false,
                //async_timer_futures: FuturesUnordered::new(),
                //current_time: Instant::now(),
                //async_delayed_prepare: None,

                // partition_delayed_msgs: Vec::new(),
                // partition_public_keys: HashSet::new(),
                simulate_asynchrony,
                asynchrony_type: async_type
                    .iter()
                    .map(|v| AsyncEffectType::from(*v))
                    .collect(),
                asynchrony_start,
                asynchrony_duration,
                affected_nodes,
                asynchrony_node_ids_per_window,
                during_simulated_asynchrony: false,
                current_effect_type: AsyncEffectType::Off,
                current_asynchrony_node_ids: Vec::new(),
                current_num_affected_nodes: 0,
                async_timer_futures: FuturesUnordered::new(),

                current_time: Instant::now(),
                // //For full delay
                async_delayed_prepare: None,
                // //For partition
                partition_public_keys: HashSet::new(),
                partition_delayed_msgs: Vec::new(),
                //For egress
                egress_penalty: resolved_egress_penalty,
                //egress_delay_queue: DelayQueue::new(),
                //delayed_messages: VecDeque::new(),
                //egress_timer_futures: FuturesUnordered::new(),
                //egress_timer: Timer::new(0, 0, egress_penalty),
                //egress_delayed_msgs: VecDeque::new(),
                egress_delay_queue: DelayQueue::new(),
                current_egress_end: Instant::now(),
                use_expoential_timeouts,
                dropped_slot: 0,
                tx_worker_async_channel,
                payload_timer_futures: FuturesUnordered::new(),
                missed_payloads: 0,
                target_ip_addresses,
                last_headers: VecDeque::with_capacity(2),
                // Metrics collection parameters
                epoch_slots,
                window_size,
                last_triggered_slot: 0, // Initialize to 0
                total_committed_transactions: 0,
                last_triggered_tx_count: 0,
                epoch_slot_start: None,
                epoch_slot_end: None,
                // Channel to receive metrics state
                rx_metrics_state,
                tx_metrics_state,
                // Persistent connection to metrics_collector
                metrics_collector_sender: None,
                metrics_collector_task: None,
                state_reports: HashMap::new(),
                collection_timer_futures: FuturesUnordered::new(),
                collection_timeout_ms: timeout_delay,
                active_collection_timers: HashSet::new(),
                pending_aggregate_reports: BTreeMap::new(),
                certifiable_aggregate_reports: HashMap::new(),
                certified_aggregate_epochs: HashSet::new(),
                aggregated_epochs: HashSet::new(),
                rl_param_update_sender,
                rl_param_update_receiver,
                rl_param_update_task: None,
                // Request ID counter
                next_request_id: 0,
                // RL parameter update signal counter
                rl_param_signal_epoch: 0,
                // Slot in each epoch to apply parameter updates
                applied_begin,
                // Pending parameter update epoch
                pending_param_update_epoch: None,
                // Pending parameter update params
                pending_param_update_params: None,
                param_apply_ok: false,
                param_apply_ok_epoch: 0,
                // Channel to send parameter updates to proposer
                tx_proposer_params,
                // Parameters tracked for dynamic updates
                header_size,
                max_header_delay,
                batch_size,
                max_batch_delay,
                aggregation_strategy,
                data_pollution_node_ids,
                data_pollution_prob,
                data_pollution_strategy,
            }
            .run()
            .await;
        });
    }

    async fn process_own_header(&mut self, mut header: Header) -> DagResult<()> {
        //println!("Received own header");
        debug!(
            "Processing own header with {:?} consensus messages",
            header.consensus_messages.len()
        );
        // for (dig, consensus) in &header.consensus_messages {
        //     match consensus { //TODO: Re-factor ConsensusMessages to all have slot/view, option for TC/QC, and a type.
        //         ConsensusMessage::Prepare {slot, view, tc: _, qc_ticket: _, proposals: _, } => {debug!("Prepare instance for slot {}", slot);},
        //         ConsensusMessage::Confirm {slot, view, qc: _, proposals: _, } => {debug!("Confirm instance for slot {}", slot);},
        //         ConsensusMessage::Commit {slot, view, qc: _, proposals: _, } => {debug!("Commit instance for slot {}", slot);},
        //     };
        // }

        //GC all obsolete qc_makers //WARNING: FIXME: Can only do this here if Votes are piggybacked on cars (i.e. not external and never delayed)
        //self.qc_makers.clear();

        // Update the current header we are collecting votes for
        // Keep last headers for equivocation simulation
        self.last_headers.push_front(header.clone());
        while self.last_headers.len() > 2 {
            self.last_headers.pop_back();
        }
        self.current_header = header.clone();
        // Indicate that we haven't sent a cert yet for this header
        self.sent_cert_to_proposer = false;

        // Reset the votes aggregator.
        self.votes_aggregator = VotesAggregator::new();

        match self.use_optimistic_tips {
            //Add early here, so that enough coverage will include leader tip.
            true => self.current_proposal_tips.insert(
                header.origin(),
                Proposal {
                    header_digest: header.digest(),
                    height: header.height(),
                },
            ),
            false => self.current_certified_tips.insert(
                header.origin(),
                Proposal {
                    header_digest: header.digest(),
                    height: header.height(),
                },
            ),
        };

        // Augment consensus messages with latest prepares
        for consensus in header.consensus_messages.values_mut() {
            self.set_consensus_proposal(consensus);
        }

        //Set all consensus instances
        for (dig, consensus) in &header.consensus_messages {
            match consensus {
                //TODO: Re-factor ConsensusMessages to all have slot/view, option for TC/QC, and a type.
                ConsensusMessage::Prepare {
                    slot,
                    view,
                    tc: _,
                    qc_ticket: _,
                    proposals: _,
                    aggregate_report: _,
                } => {
                    self.consensus_instances
                        .insert((*slot, dig.clone()), consensus.clone());
                }
                ConsensusMessage::Confirm {
                    slot,
                    view,
                    qc: _,
                    proposals: _,
                    aggregate_report: _,
                } => {
                    self.consensus_instances
                        .insert((*slot, dig.clone()), consensus.clone());
                }
                _ => {}
            };
            //self.consensus_instances.insert(dig.clone(), consensus.clone());
        }

        self.send_msg(
            PrimaryMessage::Header(header.clone(), false),
            header.height,
            None,
            false,
        )
        .await;

        // Process the header.
        self.process_header(header, false).await
    }

    async fn create_fake_header(&mut self, original: &Header, partition_id: u8) -> Header {
        use config::WorkerId;
        use crypto::Digest;
        use std::collections::BTreeMap;

        // Create fake payload with same size but different content based on partition
        let mut fake_payload: BTreeMap<Digest, WorkerId> = BTreeMap::new();
        for (i, (_, worker_id)) in original.payload.iter().enumerate() {
            // Generate a different digest based on partition_id and index
            let mut fake_digest_bytes = [0u8; 32];
            fake_digest_bytes[0..8].copy_from_slice(&(i as u64).to_le_bytes());
            fake_digest_bytes[8] = partition_id;
            fake_digest_bytes[12..20].copy_from_slice(&original.digest().0[12..20]);
            fake_digest_bytes[20..28]
                .copy_from_slice(&(original.height + partition_id as u64).to_le_bytes());
            fake_digest_bytes[28..32].copy_from_slice(&original.digest().0[28..32]);
            fake_payload.insert(Digest(fake_digest_bytes), *worker_id);
        }

        // Keep same consensus_messages structure
        let fake_consensus = original.consensus_messages.clone();

        debug!(
            "Creating fake header for partition {} with {} payload items (original had {})",
            partition_id,
            fake_payload.len(),
            original.payload.len()
        );

        Header::new(
            original.author,
            original.height,
            fake_payload,
            BTreeMap::new(), // fake batch_metadata
            original.parent_cert.clone(),
            &mut self.signature_service,
            fake_consensus,
            original.num_active_instances,
        )
        .await
    }

    fn is_malicious_node(&self) -> bool {
        // Prefer explicit per-window targets when provided.
        let mut targeted = false;
        if !self.current_asynchrony_node_ids.is_empty() {
            targeted = self.current_asynchrony_node_ids.contains(&self.node_index);
            debug!(
                "Node {} (idx {}) targeted by explicit async node ids {:?}: {}",
                self.name, self.node_index, self.current_asynchrony_node_ids, targeted
            );
        }
        return targeted;

        // Fallback: legacy index-based malicious selection.
        // let mut keys: Vec<_> = self.committee.authorities.keys().cloned().collect();
        // keys.sort();
        // let node_index = keys.binary_search(&self.name).unwrap_or(0);
        // let f = (self.committee.size() - 1) / 3;
        // let max_malicious = f;

        // let is_malicious = node_index < max_malicious as usize;
        // debug!("Node {} (index {}) is malicious: {} (max_malicious: {})",
        //        self.name, node_index, is_malicious, max_malicious);
        // is_malicious
    }

    #[async_recursion]
    async fn process_header(&mut self, header: Header, sync: bool) -> DagResult<()> {
        debug!("Processing Header:  {:?}", header);
        debug!("Processing the header with height {:?}", header.height);

        // Check the parent certificate. Ensure the certificate contains a quorum of votes and is
        // at the preivous height
        let stake: Stake = header
            .parent_cert
            .votes
            .iter()
            .map(|(pk, _)| self.committee.stake(pk))
            .sum();
        //println!("Before first ensure");
        debug!("Past header parent cert stake check");
        ensure!(
            header.parent_cert.height() + 1 == header.height(),
            DagError::MalformedHeader(header.id.clone())
        );
        debug!("Past header parent cert height check");

        //println!("Before second ensure");
        ensure!(
            stake >= self.committee.validity_threshold() || header.parent_cert.height() == 0,
            DagError::HeaderRequiresQuorum(header.id.clone())
        );
        debug!("Past header parent cert stake check");
        //println!("After second ensure");

        // Process the parent certificate
        self.process_certificate(header.clone().parent_cert).await?;

        // Ensure we have the payload. If we don't, the synchronizer will ask our workers to get it, and then
        // reschedule processing of this header once we have it.
        if self.synchronizer.missing_payload(&header, sync).await? {
            //println!("Missing payload");
            debug!("Processing of {} suspended: missing payload", header);
            /*let timer = PayloadTimer::new(header.clone(), 5000);
            self.payload_timer_futures.push(Box::pin(timer));*/
            return Ok(());
        }

        // Write this header as an optimistic tip
        if self.use_optimistic_tips {
            debug!("Wrote optimistic tip to store");
            let mut optimistic_key = header.digest().to_vec();
            optimistic_key.push(1);
            debug!("optimistic tip length vector is {}", optimistic_key.len());
            debug!("process header optimistic key is {:?}", optimistic_key);
            let dummy_vec: Vec<u8> = vec![1];
            self.store.write(optimistic_key, dummy_vec).await;

            /*match self.store.read(optimistic_key.clone()).await? {
                Some(dummy_value) => {
                    debug!("can read our written optimistic key {:?}, dummy val is {:?}", optimistic_key, dummy_value);
                },
                None => { debug!("cannot read our own optimistic key {:?}", optimistic_key); },
            }*/
        }

        // By FIFO should have parent of this header (and recursively all ancestors), reschedule for processing if we don't
        if self
            .synchronizer
            .get_parent_header(&header)
            .await?
            .is_none()
        {
            //println!("The parent is missing");
            debug!("The parent is missing, suspending processing");
            return Ok(());
        }

        // Check whether we can seamlessly vote for all consensus messages, if not reschedule
        if !self.is_consensus_ready(&header).await {
            // TODO: Keep track of stats of sync
            // NOTE: This blocks if prepare tips are not available, the leader of the prepare takes
            // on the responsibility of possible blocking i.e. its lane won't continue
            // TODO: Use reputation
            //println!("Need to sync on missing tips, reschedule");
            debug!("Can't vote for prepare, need to sync on missing tips, suspending processing");
            return Ok(());
        }

        //println!("storing the header");
        debug!("storing the header");

        // Store the header since we have the parents (recursively).
        let bytes = bincode::serialize(&header).expect("Failed to serialize header");
        self.store.write(header.digest().to_vec(), bytes).await;

        // If the header received is at a greater height then add it to our local tips and proposals
        if self.use_optimistic_tips
            && header.height()
                > self
                    .current_proposal_tips
                    .get(&header.origin())
                    .unwrap()
                    .height
        {
            self.current_proposal_tips.insert(
                header.origin(),
                Proposal {
                    header_digest: header.digest(),
                    height: header.height(),
                },
            );
            //println!("updating tip");
            debug!("updating tip");

            // Since we received a new tip, check if any of our pending tickets are ready
            self.try_prepare_waiting_slots().await?;
        }

        //println!("after height check");
        debug!("after tip height check");

        // If Header has no consensus messages (i.e. is pure car) then all replicas can vote
        // This ensures that all nodes can form certificates and continue proposing headers
        // Original logic: only 2f+1 replicas need to vote (f nodes suppressed)
        // Modified: all nodes vote to prevent lane stagnation
        // if header.consensus_messages.is_empty() && !self.check_cast_vote(&header) {
        //     return Ok(());
        // }

        // Check if we can vote for this header.
        if self
            .last_voted
            .entry(header.height())
            .or_insert_with(HashSet::new)
            .insert(header.author)
        {
            //println!("voting for header");
            // Process the consensus instances contained in the header (if any)
            let consensus_votes = self.process_consensus_messages(&header).await?;

            //println!("Consensus sigs length {:?}", consensus_votes.len());
            debug!("Consensus sigs length {:?}", consensus_votes.len());

            // Create a vote for the header and any valid consensus instances
            let vote = Vote::new(
                &header,
                &self.name,
                &mut self.signature_service,
                consensus_votes,
            )
            .await;
            //println!("Created vote");
            debug!("Created Vote {:?}", vote);

            if vote.origin == self.name {
                self.process_vote(vote, false)
                    .await
                    .expect("Failed to process our own vote");
            } else {
                /*let address = self
                    .committee
                    .primary(&header.author)
                    .expect("Author of valid header is not in the committee")
                    .primary_to_primary;
                let bytes = bincode::serialize(&PrimaryMessage::Vote(vote))
                    .expect("Failed to serialize our own vote");
                let handler = self.network.send(address, Bytes::from(bytes)).await;
                self.cancel_handlers
                    .entry(header.height())
                    .or_insert_with(Vec::new)
                    .push(handler);*/

                self.send_msg(
                    PrimaryMessage::Vote(vote),
                    header.height(),
                    Some(header.author),
                    false,
                )
                .await;
            }
        }
        Ok(())
    }

    fn check_cast_vote(&self, header: &Header) -> bool {
        //Only 2f+1 replicas need to vote for cars; i.e. skip f //Alternatively: Consider yourself a voter if name within 2f+1 after author
        let mut start = false;
        let mut count = 1; //start at 1, f do not need to vote.

        let mut keys: Vec<_> = self.committee.authorities.keys().cloned().collect();
        keys.sort();
        debug!("Committee iteration order: {:?}", keys);
        debug!("Author: {:?}, My name: {:?}", header.author, self.name);

        let mut iter = self.committee.authorities.iter();
        let mut iteration_order = Vec::new();
        let mut author_position = None;
        let mut my_position = None;
        let mut suppressed_nodes = Vec::new();

        //find origin position. After that identify first f that should not send.
        while count < self.committee.validity_threshold() {
            let x = iter.next();
            if x.is_none() {
                iter = self.committee.authorities.iter(); //wrap around
                continue;
            }
            let (id, _) = x.unwrap();
            iteration_order.push(id.clone());

            if header.author.eq(&id) {
                start = true;
                author_position = Some(iteration_order.len() - 1);
                debug!(
                    "Found author at iteration position: {}",
                    iteration_order.len() - 1
                );
                continue;
            }
            if start {
                if self.name.eq(id) {
                    my_position = Some(iteration_order.len() - 1);
                    let suppressed_index = keys.binary_search(id).unwrap_or(usize::MAX);
                    let author_index = keys.binary_search(&header.author).unwrap_or(usize::MAX);
                    debug!(
                        "DO NOT CAST VOTE for header: {}, suppressed_pk={:?}, suppressed_index={}, author_index={}, iteration_pos={}",
                        header.id,
                        id,
                        suppressed_index,
                        author_index,
                        iteration_order.len() - 1
                    );
                    return false;
                }
                suppressed_nodes.push(id.clone());
                count += 1;
            }
        }

        debug!("Iteration order: {:?}", iteration_order);
        debug!(
            "Author position: {:?}, My position: {:?}",
            author_position, my_position
        );
        debug!("Suppressed nodes: {:?}", suppressed_nodes);
        debug!("CAST VOTE for header: {}", header.id);
        return true;

        //Alternatively: Count 2f+1 that should send.
        //let mut count = 0;
        // while count < self.committee.quorum_threshold() {
        //     let x = iter.next();
        //     if x.is_none(){
        //         iter = self.committee.authorities.iter(); //wrap around
        //         continue;
        //     }
        //     let (id, _) = x.unwrap();
        //     if header.author.eq(&id) {
        //         start = true;
        //     }
        //     if start {
        //         if self.name.eq(id) {
        //             debug!("CAST VOTE for header: {}", header.id);
        //             return true;
        //         }
        //         count += 1;
        //     }
        // }
        // debug!("DO NOT CAST VOTE for header: {}", header.id);
        // return false;
    }

    #[async_recursion]
    async fn process_vote(&mut self, vote: Vote, is_loopback: bool) -> DagResult<()> {
        debug!("Processing Vote {:?}", vote);

        // NOTE: If sending externally then need map of open consensus instances

        //If consensus vote loopback => Look up digest directly instead of via current instance.
        let consensus_loopback = is_loopback && !vote.consensus_votes.is_empty(); //vote.consensus_instance.is_some();

        // Only process votes for the current header (or loopbacks for consensus)
        if vote.id != self.current_header.id || consensus_loopback {
            //println!("Wrong header");
            return Ok(());
        }

        if self.is_malicious_node() && self.current_header.author == self.name {
            debug!(
                "MALICIOUS NODE received vote: header_id={:?}, height={}, from_author={:?}, vote_author={:?}",
                vote.id,
                vote.height,
                vote.author,
                vote.author
            );
        }

        // debug!(
        //     "received vote for our header: header_id={:?}, height={}, from_author={:?}",
        //     vote.id,
        //     vote.height,
        //     vote.author
        // );

        //Invariant: All votes contain the same content (i.e. it's not the case that some of them carry things like timeouts etc)
        //Wait to form num_active instance many QCs

        //TODO: continue earlier if timeouts expire!! Currently all our lanes will stop if consensus stops voting
        //Car should still vote even if consensus says No.

        let num_active_consensus_messages = self.current_header.num_active_instances;
        debug!("num active instances {:?}", num_active_consensus_messages);

        // Iterate through vote for each consensus instance
        for (slot, digest, sig) in vote.consensus_votes.iter() {
            debug!("current header {:?}", self.current_header);
            debug!("digest is {:?}", digest);
            //Get vote type of the instance: Prepare/Confirm-vote

            let opt_curr_instance = self.consensus_instances.get(&(*slot, digest.clone()));
            if opt_curr_instance.is_none() {
                debug!("consensus instance slot has committed, skip processing vote");
                continue;
            }
            let current_instance = opt_curr_instance.unwrap();

            if !is_loopback && vote.author != self.name {
                //Verify signature. Could optimize performance by only verifying after forming a batch, and use parallel batch_verification
                sig.verify(&current_instance.digest(), &vote.author)?;
            }
            //Why does this code not work?
            //let current_instance = self.consensus_instances.get(&(*slot, digest.clone())).unwrap(); //todo: Throw a panic if it does not exist.

            // let current_instance = match consensus_loopback {
            //     true => &vote.consensus_instance.as_ref().unwrap(), //Just look it up from the buffered instance
            //     false => self.current_header.consensus_messages.get(digest).unwrap(),
            // };

            let qc_maker = self
                .qc_makers
                .entry((*slot, digest.clone()))
                .or_insert(QCMaker::new());
            // let qc_maker = match current_instance {
            //     ConsensusMessage::Prepare {slot, view, tc: _, proposals: _, } => self.qc_makers.entry((*slot, digest.clone())).or_insert(QCMaker::new()), //self.pqc_makers.entry((*slot, *view)).or_insert(QCMaker::new()),
            //     ConsensusMessage::Confirm {slot, view, qc: _, proposals: _, } => self.qc_makers.entry((*slot, digest.clone())).or_insert(QCMaker::new()), //self.cqc_makers.entry((*slot, *view)).or_insert(QCMaker::new()),
            //     _ => unreachable!("Should never try and fetch a qc_maker for Commit"),
            // };

            //    // If not already a qc maker for this consensus instance message, create one
            //     match self.qc_makers.get(&digest) {
            //         Some(_) => {
            //             //println!("QC Maker already exists");
            //         }
            //         None => {
            //             self.qc_makers.insert(digest.clone(), QCMaker::new());
            //         }
            //     }

            //     // Otherwise get the qc maker for this instance
            //     let qc_maker = self.qc_makers.get_mut(&digest).unwrap();

            //Configure qc_maker to try to use Fast Path
            qc_maker.try_fast = match current_instance {
                ConsensusMessage::Prepare {
                    slot: _,
                    view: _,
                    tc: _,
                    qc_ticket: _,
                    proposals: _,
                    aggregate_report: _,
                } => self.use_fast_path && !qc_maker.fast_path_disabled, //Only PrepareQC should try to compute a FastQC
                _ => false,
            };

            //println!("qc maker weight {:?}", qc_maker.votes.len());

            // Add vote to qc maker, if a QC forms then create a new consensus instance
            // TODO: Put fast path logic in qc maker (decide whether to wait timeout etc.), add
            // external messages

            //If qc_ready, but qc_opt = None => This is first Slow QC;
            //If qc_ready and qc_opt => This is FastQC or Consumption of Loopback to fetch SlowQC
            let (qc_ready, qc_opt) = match is_loopback {
                false => {
                    qc_maker.append(vote.author, (digest.clone(), sig.clone()), &self.committee)?
                }
                true => {
                    qc_maker.try_fast = false; //turn back to normal path handling
                    qc_maker.fast_path_disabled = true;
                    qc_maker.get_qc()?
                }
            };

            if qc_ready {
                // if let Some(qc) = qc_maker.append(vote.author, (digest.clone(), sig.clone()), &self.committee)?
                // {
                if qc_opt.is_none() && self.use_fast_path {
                    // Slow QC is available but we should wait for Fast
                    //Start timer for Fast:
                    //Creates a dummy vote with the same id as this vote, but only the waiting digest as consensus sigs
                    //Upon triggering timer, it will call loopback again, which will get the QC and proceed.
                    //By including only the digest of the missing instance we avoid duplicates.
                    //Alternatively could modify QCMaker such that it wipes the QC after first use

                    let t_vote = Vote {
                        id: Digest::default(), //vote.id.clone(),
                        height: 0,
                        origin: PublicKey::default(),
                        author: PublicKey::default(),
                        signature: Signature::default(),
                        consensus_votes: vec![(*slot, digest.clone(), Signature::default())],
                        //consensus_instance: Some(current_instance.clone()), //Buffer instance. Current header could've advanced in the meantime and thus no longer include this instance by the time timer triggers
                    };
                    let fast_timer = CarTimer::new(t_vote, self.fast_path_timeout);
                    self.car_timer_futures.push(Box::pin(fast_timer));
                    //self.timers.insert((tc.slot, tc.view + 1));
                } else if let Some(qc) = qc_opt {
                    //If QC = some (i.e. FastPathQC succeed, or SlowPathQC suceed if running without FP)
                    //println!("QC formed");
                    self.current_qcs_formed += 1;

                    // let current_instance = self
                    //     .current_header
                    //     .consensus_messages
                    //     .get(&digest)
                    //     .unwrap();
                    match current_instance {
                        ConsensusMessage::Prepare {
                            slot,
                            view,
                            tc: _,
                            qc_ticket: _,
                            proposals,
                            aggregate_report,
                        } => {
                            debug!("Prepare QC formed in slot {:?}", slot);
                            debug!(
                                "Prepare has slot: {}, view: {}, digest: {}",
                                slot,
                                view,
                                current_instance.digest()
                            );

                            //TODO: FIXME: (I assume this is the leader tip optimization): Re-factor this to be set at Header propose time already.
                            // Create a tip proposal for the header which contains the prepare message, so that it can be committed as part of the proposals
                            /*let leader_tip_proposal: Proposal = Proposal {header_digest: self.current_header.digest(), height: self.current_header.height(),};
                            // Add this cert to the proposals for this instance
                            let mut new_proposals = proposals.clone();
                            new_proposals.insert(self.name, leader_tip_proposal);*/

                            let new_consensus_message = match qc_maker.try_fast {
                                true => {
                                    debug!("taking fast path!");

                                    ConsensusMessage::Commit {
                                        slot: *slot,
                                        view: *view,
                                        qc,
                                        proposals: proposals.clone(),
                                        aggregate_report: aggregate_report.clone(),
                                    }
                                } // Create Commit if we have FastPrepareQC
                                false => {
                                    debug!("taking slow path!");

                                    ConsensusMessage::Confirm {
                                        slot: *slot,
                                        view: *view,
                                        qc,
                                        proposals: proposals.clone(),
                                        aggregate_report: aggregate_report.clone(),
                                    }
                                }
                            };
                            //let new_consensus_message = ConsensusMessage::Confirm {slot: *slot, view: *view,  qc, proposals: new_proposals,};

                            // Send this new instance to the proposer
                            self.tx_info
                                .send(new_consensus_message)
                                .await
                                .expect("Failed to send info");
                        }
                        ConsensusMessage::Confirm {
                            slot,
                            view,
                            qc: _,
                            proposals,
                            aggregate_report,
                        } => {
                            debug!("Commit QC formed in slot {:?}", slot);

                            let new_consensus_message = ConsensusMessage::Commit {
                                slot: *slot,
                                view: *view,
                                qc,
                                proposals: proposals.clone(),
                                aggregate_report: aggregate_report.clone(),
                            };

                            // Send this new instance to the proposer
                            self.tx_info
                                .send(new_consensus_message)
                                .await
                                .expect("Failed to send info");
                        }
                        ConsensusMessage::Commit {
                            slot: _,
                            view: _,
                            qc: _,
                            proposals: _,
                            aggregate_report: _,
                        } => {}
                    };
                }
            }
        }

        // If there are some consensus instances in the header then wait for 2f+1 votes to form QCs
        //let consensus_ready: bool = !self.current_header.consensus_messages.is_empty() && self.current_qcs_formed == num_active_consensus_messages;
        //NEW: Consider consensus ready if there is nothing we need to wait for either!
        let consensus_ready: bool = self.current_header.consensus_messages.is_empty()
            || self.current_qcs_formed == num_active_consensus_messages;

        //Next: Check whether Car is ready to go
        let vote_id = vote.id.clone();
        let car_timeout = is_loopback && vote.consensus_votes.is_empty();

        // Add the vote to the votes aggregator for the actual header
        //Note: car_cert_ready is true if QC exists (f+1 votes); first = true when QC is formed the first time (this starts timer only once)
        //=> aggregator will ignore new votes after (in particular it will ignore the fake loopback vote)
        let (car_cert_ready, first) =
            self.votes_aggregator
                .append(vote, &self.committee, &self.current_header)?;

        if self.is_malicious_node() && self.current_header.author == self.name {
            debug!(
                "MALICIOUS NODE vote aggregation: header_id={:?}, height={}, car_cert_ready={}, first={}, current_votes_count={}",
                vote_id,
                self.current_header.height,
                car_cert_ready,
                first,
                self.votes_aggregator.votes.len()
            );
        }

        //Consider consensus "ready" if we timed out (i.e. just move on without waiting for consensus)
        let consensus_ready = consensus_ready || car_timeout;
        //only take the dissemination QC if consensus is ready, or we have timed out (this avoids needless copies)
        let dissemination_cert = match car_cert_ready && consensus_ready {
            true => self.votes_aggregator.get()?, //Get will only return Cert ONCE. I.e. if timer loopbacks after it's already been used, then nothing happens.
            false => None,
        };

        //Old: If there are no consensus instances in the header then only wait for the dissemination cert (f+1) votes
        //let dissemination_ready: bool = self.current_header.consensus_messages.is_empty() && dissemination_cert.is_some();
        //New: dissemination ready as soon as
        let dissemination_ready: bool = car_cert_ready && dissemination_cert.is_some();

        debug!(
            "sentToProposer {:?}, diss_ready {:?}, consensus_ready {:?}",
            self.sent_cert_to_proposer, dissemination_ready, consensus_ready
        );

        //If ready to disseminate car (dissemination cert exists) but waiting for consensus
        if dissemination_ready && !consensus_ready && first {
            //first => start only one Timer
            let t_vote = Vote {
                id: vote_id.clone(),
                height: 0,
                origin: PublicKey::default(),
                author: PublicKey::default(),
                signature: Signature::default(),
                consensus_votes: vec![], //Create dummy vote with no sigs => this indicates its the Car timeout
                                         //consensus_instance: None
            };
            debug!("car timer starts");
            let fast_timer = CarTimer::new(t_vote, self.car_timeout);
            self.car_timer_futures.push(Box::pin(fast_timer));
        }

        //if !self.sent_cert_to_proposer && (dissemination_ready || consensus_ready) {
        if !self.sent_cert_to_proposer && (dissemination_ready && consensus_ready) {
            //debug!("Assembled {:?}", dissemination_cert.unwrap());
            //println!("diss ready {:?}, consensus ready {:?}", dissemination_ready, consensus_ready);
            debug!(
                "formed dissemination certificate for our header: header_id={:?}, height={}, sending to proposer",
                vote_id.clone(),
                self.current_header.height
            );

            if self.is_malicious_node() && self.current_header.author == self.name {
                debug!(
                    "MALICIOUS NODE formed certificate: header_id={:?}, height={}, cert_votes_count={}, cert_origin={:?}",
                    vote_id,
                    self.current_header.height,
                    dissemination_cert.as_ref().map(|c| c.votes.len()).unwrap_or(0),
                    dissemination_cert.as_ref().map(|c| c.origin()).unwrap_or(PublicKey::default())
                );
            }

            self.tx_proposer
                .send(dissemination_cert.unwrap())
                .await
                .expect("Failed to send certificate");

            self.sent_cert_to_proposer = true;
            //println!("after sending to proposer");
            self.current_qcs_formed = 0;
        }

        // TODO: Handle invalidated case where possibly want to send consensus message externally,
        // will add this when the fast path is added
        Ok(())
    }

    //TODO: Then work on Process Vote //TODO: Add a function: SendConsensus
    async fn process_consensus_vote(
        &mut self,
        vote: ConsensusVote,
        is_loopback: bool,
    ) -> DagResult<()> {
        debug!("Receive consensus vote for dig {}", &vote.digest);

        let opt_curr_instance = self
            .consensus_instances
            .get(&(vote.slot, vote.digest.clone()));
        if opt_curr_instance.is_none() {
            debug!("consensus instance slot has committed, skip processing vote");
            return Ok(());
        }

        if !is_loopback && vote.author != self.name {
            //Verify signature. Could optimize performance by only verifying after forming a batch, and use parallel batch_verification
            vote.sig.verify(&vote.digest, &vote.author)?;
        }

        let current_instance = opt_curr_instance.unwrap();
        //Invariant: All votes contain the same content (i.e. it's not the case that some of them carry things like timeouts etc)
        //Wait to form num_active instance many QCs

        let qc_maker = self
            .qc_makers
            .entry((vote.slot, vote.digest.clone()))
            .or_insert(QCMaker::new());

        //Configure qc_maker to try to use Fast Path
        qc_maker.try_fast = match current_instance {
            ConsensusMessage::Prepare {
                slot: _,
                view: _,
                tc: _,
                qc_ticket: _,
                proposals: _,
                aggregate_report: _,
            } => {
                debug!(
                    "qc_maker.fast_path_disabled {:?}",
                    qc_maker.fast_path_disabled
                );
                self.use_fast_path && !qc_maker.fast_path_disabled //Only PrepareQC should try to compute a FastQC
            }
            _ => false,
        };

        // Add vote to qc maker, if a QC forms then create a new consensus instance

        //If qc_ready, but qc_opt = None => This is first Slow QC;
        //If qc_ready and qc_opt => This is FastQC or Consumption of Loopback to fetch SlowQC
        let (qc_ready, qc_opt) = match is_loopback {
            false => qc_maker.append(
                vote.author,
                (vote.digest.clone(), vote.sig.clone()),
                &self.committee,
            )?,
            true => {
                qc_maker.try_fast = false; //turn back to normal path handling
                qc_maker.fast_path_disabled = true;
                qc_maker.get_qc()?
            }
        };

        debug!("qc_maker.try_fast {:?}", qc_maker.try_fast);
        debug!("qc_ready {:?}", qc_ready);
        debug!("qc_opt {:?}", qc_opt);

        debug!("qc maker weight {:?}", qc_maker.votes.len());

        if qc_ready {
            if qc_opt.is_none() && self.use_fast_path {
                // Slow QC is available but we should wait for Fast
                //Start timer for Fast:
                //Creates a dummy vote with the same id as this vote, but only the waiting digest as consensus sigs
                //Upon triggering timer, it will call loopback again, which will get the QC and proceed.
                //By including only the digest of the missing instance we avoid duplicates.
                //Alternatively could modify QCMaker such that it wipes the QC after first use

                let fast_timer = FastTimer::new(vote.clone(), self.fast_path_timeout);
                debug!("Fast path timer start");
                self.fast_timer_futures.push(Box::pin(fast_timer));
                //self.timers.insert((tc.slot, tc.view + 1));
            } else if let Some(qc) = qc_opt {
                //If QC = some (i.e. FastPathQC succeed, or SlowPathQC suceed if running without FP)
                //println!("QC formed");
                match current_instance {
                    ConsensusMessage::Prepare {
                        slot,
                        view,
                        tc: _,
                        qc_ticket: _,
                        proposals,
                        aggregate_report,
                    } => {
                        debug!("Prepare QC formed in slot {:?}", slot);
                        debug!(
                            "Prepare has slot: {}, view: {}, digest: {}",
                            slot,
                            view,
                            current_instance.digest()
                        );

                        match qc_maker.try_fast {
                            true => {
                                debug!("taking fast path for slot {:?}", slot);
                                if let Some(report) = aggregate_report {
                                    info!(
                                        "⚡ Fast-CC ready for Prepare(slot={}, view={}) with AggregateReport digest {}. Broadcasting COMMIT",
                                        slot, view, report.digest()
                                    );
                                }

                                let new_consensus_message = ConsensusMessage::Commit {
                                    slot: *slot,
                                    view: *view,
                                    qc,
                                    proposals: proposals.clone(),
                                    aggregate_report: aggregate_report.clone(),
                                };
                                self.send_consensus_req(new_consensus_message).await?;
                            } // Create Commit if we have FastPrepareQC
                            false => {
                                let sent_confirm = qc_maker.sent_confirm.clone();
                                let completed_fast = qc_maker.completed_fast;
                                if !sent_confirm && !completed_fast {
                                    debug!("sending confirm for slot {:?} with qc {:?}", slot, qc);
                                    debug!("taking slow path for slot {:?}", slot);

                                    if let Some(report) =
                                        self.certifiable_aggregate_reports.get(&(*slot, *view))
                                    {
                                        info!(
                                            "🐢 Slow path entered for epoch {} at Prepare(slot={}, view={}) with AggregateReport digest {}. Broadcasting CONFIRM",
                                            report.epoch, slot, view, report.digest()
                                        );
                                    }
                                    let new_consensus_message = ConsensusMessage::Confirm {
                                        slot: *slot,
                                        view: *view,
                                        qc,
                                        proposals: proposals.clone(),
                                        aggregate_report: aggregate_report.clone(),
                                    };
                                    qc_maker.sent_confirm = true;
                                    self.send_consensus_req(new_consensus_message).await?;
                                }
                            }
                        };
                        //let new_consensus_message = ConsensusMessage::Confirm {slot: *slot, view: *view,  qc, proposals: new_proposals,};

                        // continue with next consensus phase
                    }
                    ConsensusMessage::Confirm {
                        slot,
                        view,
                        qc: _,
                        proposals,
                        aggregate_report,
                    } => {
                        debug!("Commit QC formed in slot {:?}", slot);
                        if let Some(report) =
                            self.certifiable_aggregate_reports.get(&(*slot, *view))
                        {
                            info!(
                                "🐢 Slow-CC ready for epoch {} at slot {} view {} with AggregateReport digest {}. Broadcasting COMMIT",
                                report.epoch, slot, view, report.digest()
                            );
                        }

                        let new_consensus_message = ConsensusMessage::Commit {
                            slot: *slot,
                            view: *view,
                            qc,
                            proposals: proposals.clone(),
                            aggregate_report: aggregate_report.clone(),
                        };

                        // Send this new instance to the proposer
                        self.send_consensus_req(new_consensus_message).await?;
                    }
                    ConsensusMessage::Commit {
                        slot: _,
                        view: _,
                        qc: _,
                        proposals: _,
                        aggregate_report: _,
                    } => {
                        panic!("Should never receive Vote for Commit")
                    }
                };
            }
        }

        Ok(())
    }

    fn set_consensus_proposal(&mut self, consensus_message: &mut ConsensusMessage) {
        let header = &self.current_header;
        match consensus_message {
            //TODO: Re-factor ConsensusMessages to all have slot/view, option for TC/QC, and a type.
            ConsensusMessage::Prepare {
                slot,
                view,
                tc,
                qc_ticket: _,
                proposals,
                aggregate_report: _,
            } => {
                let set_proposal = tc.is_none() || proposals.is_empty();
                //Set tips to propose if it is a new proposal (empty by default), or winning_proposal = empty. => if there is a winning prop proposals will not be empty
                if set_proposal {
                    debug!("UPDATING HEADER for slot {}", slot);
                    // Add new proposal tips

                    *proposals = match self.use_optimistic_tips {
                        true => self.current_proposal_tips.clone(),
                        false => self.current_certified_tips.clone(),
                    };

                    // Leader tip proposal
                    proposals.insert(
                        self.name,
                        Proposal {
                            header_digest: header.id.clone(),
                            height: header.height,
                        },
                    );

                    for (pk, proposal) in proposals {
                        debug!("new proposal height is {:?}", proposal.height);
                    }

                    //TODO: If we want to hash also the proposals, then stored digest must change!! => have to remove entry from map and add it back with a new hash.
                }
            }
            _ => {}
        };
    }

    #[async_recursion]
    async fn send_consensus_req(
        &mut self,
        mut consensus_message: ConsensusMessage,
    ) -> DagResult<()> {
        self.set_consensus_proposal(&mut consensus_message);

        // match &consensus_message {
        //     ConsensusMessage::Prepare {slot, view, tc, qc_ticket: _, proposals} => {
        //         if self.during_simulated_asynchrony {
        //             debug!("Simulating Asynchrony: skip sending Prepare for slot {} view {}. This will trigger a view change", slot, view);
        //             self.async_delayed_prepare = Some(consensus_message);
        //             return Ok(());
        //         }
        //         // if *slot == 5 && *view == 1 {
        //         //     debug!("skip sending Prepare for slot 5 view 1. Trigger view change");
        //         //     return Ok(());
        //         // }
        //     },
        //     ConsensusMessage::Confirm {slot, view, qc: _, proposals} => {
        //         // if *slot == 5 && *view == 1 {
        //         //     debug!("skip sending Confirm for slot 5 view 1. Trigger view change");
        //         //     return Ok(());
        //         // }
        //     },
        //     ConsensusMessage::Commit {slot, view, qc: _, proposals} => {
        //         // if *slot == 5 && *view <3  {
        //         //     debug!("skip sending Commit for slot 5 view 1. Trigger view change");
        //         //     return Ok(());
        //         // }
        //     },
        // };

        debug!("Send req for Consensus message {}", consensus_message);

        let consensus_req =
            ConsensusRequest::new(self.name, consensus_message, &mut self.signature_service).await;

        //send to all others
        /*let addresses = self
            .committee
            .others_primaries(&self.name)
            .iter()
            .map(|(_, x)| x.primary_to_primary)
            .collect();
        let message = bincode::serialize(&PrimaryMessage::ConsensusRequest(consensus_req.clone())).expect("Failed to serialize timeout message");
        let handlers = self.network.broadcast(addresses, Bytes::from(message)).await;

        self.cancel_handlers
            .entry(self.current_header.height())
            .or_insert_with(Vec::new)
            .extend(handlers);*/

        self.send_msg(
            PrimaryMessage::ConsensusRequest(consensus_req.clone()),
            self.current_header.height(),
            None,
            false,
        )
        .await;

        //process oneself
        self.process_consensus_request(consensus_req).await?;

        Ok(())
    }

    #[async_recursion]
    async fn process_certificate(&mut self, certificate: Certificate) -> DagResult<()> {
        debug!("Processing {:?}", certificate);

        if certificate.origin() == self.name {
            debug!(
                "received certificate for our own header: origin={:?}, height={}, header_digest={:?}",
                certificate.origin(),
                certificate.height,
                certificate.header_digest
            );
        }

        // Store the certificate.
        let bytes = bincode::serialize(&certificate).expect("Failed to serialize certificate");
        self.store.write(certificate.digest().to_vec(), bytes).await;

        //TODO: Fix check coverage as well. (new proposals..)
        if certificate.height
            > self
                .current_certified_tips
                .get(&certificate.origin())
                .unwrap()
                .height
        {
            debug!(
                "updating tip from author {:?}, from height {:?} to height {:?}",
                certificate.origin(),
                self.current_certified_tips
                    .get(&certificate.origin())
                    .unwrap(),
                certificate.height
            );
            self.current_certified_tips.insert(
                certificate.origin(),
                Proposal {
                    header_digest: certificate.header_digest.clone(),
                    height: certificate.height,
                    //TODO: WE should also be including the cert itself (In case other replicas don't have it; so we can convince them this proposal tip is certified!)
                },
            );
            //println!("updating tip");

            // Since we received a new tip, check if any of our pending tickets are ready
            self.try_prepare_waiting_slots().await?;
        }

        //println!("Stored the certificate: {:?}", certificate.digest());

        // If we receive a new certificate from ourself, then send to the proposer, so it can make
        // a new header
        // TODO: For certified tips need to keep this check and add to list of certified tips
        /*let latest_tip = self.current_proposal_tips.get(&certificate.origin()).unwrap();
        if certificate.origin() == self.name && certificate.height() == latest_tip.height - 1 {
            //println!("Sending to proposer");
            // Send it to the `Proposer`.
            self.tx_proposer
                .send(certificate.clone())
                .await
                .expect("Failed to send certificate");
        }*/

        //println!("Certificate is {:?}, {:?}", certificate.header_digest, certificate.height);
        Ok(())
    }

    #[async_recursion]
    async fn try_prepare_waiting_slots(&mut self) -> DagResult<()> {
        //Could there even be multiple prepares? Bounding l <= 4 should make it so that each replica can only be the original leader for one slot? VC leaders are not blocked on coverage (they just propose current tips)
        debug!("prepare tickets {:?}", self.prepare_tickets);
        for i in 0..self.prepare_tickets.len() {
            //println!("checking prepare ticket");
            // Get the first buffered prepare ticket
            let prepare_msg = self.prepare_tickets.pop_front().unwrap();
            self.is_prepare_ticket_ready(&prepare_msg).await?;
        }

        Ok(())
    }

    // if !self.is_prepare_ticket_ready(prepare_message).await.unwrap() {
    //     //println!("prepare ticket not ready");
    //     self.prepare_tickets.push_back(prepare_message.clone());
    // }

    async fn is_prepare_ticket_ready(
        &mut self,
        prepare_message: &ConsensusMessage,
    ) -> DagResult<()> {
        match prepare_message {
            ConsensusMessage::Prepare {
                slot,
                view: _,
                tc: _,
                qc_ticket: _,
                proposals,
                aggregate_report: _,
            } => {
                // Ticket for next_slot. Every replica (not just the next leader) must
                // observe the same "slot is startable" condition so view-1 timeouts
                // do not begin while we are still waiting for cut-condition coverage.
                let next_slot = *slot + 1;
                let ticket_k = self.effective_k_for_prepare_slot(next_slot);

                // Bound open instances: wait for next_slot - k to commit.
                if next_slot > ticket_k {
                    debug!("beyond init k for slot {}", *slot);
                    if !self.committed_slots.contains_key(&(next_slot - ticket_k)) {
                        debug!("too many instances open");
                        self.prepare_tickets.push_back(prepare_message.clone());
                        return Ok(());
                    }
                }

                if !self.enough_coverage(&proposals) {
                    debug!(
                        "adding prepare ticket to queue for message {:?}",
                        prepare_message
                    );
                    self.prepare_tickets.push_back(prepare_message.clone());
                    return Ok(());
                }

                debug!("have enough coverage to start slot {}", next_slot);
                if !self.committed_slots.contains_key(&next_slot)
                    && !self.timers.contains(&(next_slot, 1))
                {
                    debug!("start timer for slot {} view 1", next_slot);
                    self.timer_futures
                        .push(Box::pin(Timer::new(next_slot, 1, self.timeout_delay)));
                    self.timers.insert((next_slot, 1));
                }

                let next_leader = self.leader_elector.get_leader(next_slot, 1);
                if self.name != next_leader {
                    debug!("not the next leader");
                    return Ok(());
                }

                if self.already_proposed_slots.contains(&next_slot) {
                    debug!("already proposed for slot {}", next_slot);
                    return Ok(());
                }

                let qc_ticket = match next_slot > ticket_k {
                    true => Some(
                        self.committed_slots
                            .get(&(next_slot - ticket_k))
                            .unwrap()
                            .clone(),
                    ),
                    false => None,
                };

                let expected_epoch = self.epoch_index_for_slot(next_slot);
                let aggregate_report = self.get_pending_aggregate_report(expected_epoch);
                if let Some(report) = aggregate_report.as_ref() {
                    info!(
                        "📦 Embedding AggregateReport digest {} into Prepare(slot={}, view=1)",
                        report.digest(),
                        next_slot
                    );
                }

                let new_prepare_instance = ConsensusMessage::Prepare {
                    slot: next_slot,
                    view: 1,
                    tc: None,
                    qc_ticket,
                    proposals: HashMap::new(),
                    aggregate_report,
                };

                self.already_proposed_slots.insert(next_slot);

                if self.use_ride_share {
                    self.tx_info
                        .send(new_prepare_instance)
                        .await
                        .expect("failed to send info to proposer");
                } else {
                    debug!("enough coverage!");
                    self.send_consensus_req(new_prepare_instance).await?;
                }

                Ok(())
            }
            _ => Ok(()),
        }
    }

    // TODO: Double check these checks are good enough
    #[async_recursion]
    async fn is_valid(&mut self, consensus_message: &ConsensusMessage) -> bool {
        match consensus_message {
            ConsensusMessage::Prepare {
                slot,
                view,
                tc,
                qc_ticket,
                proposals,
                aggregate_report,
            } => {
                // NOTE: There are two cases: view = 1, and view > 1
                // For view = 1 the leader can propose "anything", coverage is
                // enforced on a best effort basis
                // For view > 1, the leader must justify its prepare message with
                // a TC from the previous view, so that proposals that could have committed
                // are recovered
                let mut ticket_valid: bool = true;
                match tc {
                    Some(tc) => {
                        // Ensure tc is valid
                        if tc.view + 1 != *view {
                            return false;
                        }
                        ticket_valid = tc.verify(&self.committee).is_ok();

                        let winning_proposals = tc.get_winning_proposals(&self.committee);
                        if !winning_proposals.is_empty() {
                            for (pk, proposal) in proposals {
                                ticket_valid = ticket_valid
                                    && proposal.eq(winning_proposals.get(&pk).unwrap());
                            }
                        }
                    }
                    None => {
                        // Any prepare is valid for view 1 //TODO: Add option for sequential ticket enforcement + bounding
                        if !self.use_parallel_proposals {
                            panic!("Parallel proposals should be true");
                        }
                        // Ticket distance is taken from the message itself so in-flight
                        // prepares opened under old k remain valid while pending_k drains.
                        let max_ticket_k = self.max_allowed_ticket_k();
                        match qc_ticket.as_ref() {
                            Some(commit_qc) => {
                                let implied_k = slot.saturating_sub(commit_qc.slot);
                                debug!(
                                    "Checking QC Ticket: prepare_slot={}, qc_slot={}, implied_k={}, max_k={}",
                                    slot, commit_qc.slot, implied_k, max_ticket_k
                                );
                                if implied_k == 0 || implied_k > max_ticket_k {
                                    debug!(
                                        "QC Ticket slot mismatch: qc_slot={} + implied_k={} != prepare_slot={} (max_k={})",
                                        commit_qc.slot, implied_k, slot, max_ticket_k
                                    );
                                    return false;
                                }
                                if !self.committed_slots.contains_key(&commit_qc.slot) {
                                    debug!("Verify QC Ticket");
                                    let commit_message = transform_commitQC(commit_qc.clone());
                                    ticket_valid = self.is_valid(&commit_message).await;
                                    debug!("Verify QC Ticket: {}", ticket_valid);
                                }
                            }
                            None => {
                                if *slot > max_ticket_k {
                                    debug!(
                                        "Missing qc_ticket for Prepare slot {} requiring ticket with max_k={}",
                                        slot, max_ticket_k
                                    );
                                    return false;
                                }
                            }
                        }
                        ticket_valid = ticket_valid && *view == 1;
                    }
                };

                if let Some(report) = aggregate_report {
                    let expected_epoch = self.epoch_index_for_slot(*slot);
                    let digest = report.digest();
                    if let Err(e) = report.verify(&self.committee, expected_epoch) {
                        debug!(
                            "Prepare slot {} view {} AggregateReport verify failed: {}",
                            slot, view, e
                        );
                        return false;
                    }
                    self.certifiable_aggregate_reports
                        .insert((*slot, *view), report.clone());
                    if !self.certified_aggregate_epochs.contains(&expected_epoch) {
                        self.pending_aggregate_reports
                            .entry(expected_epoch)
                            .or_insert_with(|| report.clone());
                    }

                    // AggregateReport is advisory metadata for scheduling/telemetry.
                    // Do not reject consensus safety path when local-report inclusion differs.
                    if let Some(local_report) = self
                        .state_reports
                        .get(&expected_epoch)
                        .and_then(|reports| reports.get(&self.name))
                    {
                        let has_local_report = report.reports.iter().any(|r| {
                            r.author == self.name
                                && r.epoch == local_report.epoch
                                && r.digest() == local_report.digest()
                        });
                        if !has_local_report {
                            debug!(
                                    "Prepare slot {} view {} AggregateReport {} does not include local StateReport digest {}",
                                    slot,
                                    view,
                                    digest,
                                    local_report.digest()
                                );
                        }
                    }

                    // Seeing a valid AggregateReport is not enough to complete
                    // coordination. Keep an epoch-level candidate available for
                    // future views until a CommitQC persists the certificate.
                }

                let curr_view = self.views.get(slot).unwrap_or(&0);
                if curr_view < view {
                    self.views.insert(*slot, *view);
                }

                // Ensure that we haven't already voted in this slot, view, that the ticket is
                // valid, and we are in the same view
                !self.last_voted_consensus.contains(&(*slot, *view))
                    && ticket_valid
                    && self.views.get(slot).unwrap() == view
            }
            ConsensusMessage::Confirm {
                slot,
                view,
                qc,
                proposals: _,
                aggregate_report: _,
            } => {
                debug!("try to unwrap slot");

                let curr_view = self.views.get(slot).unwrap_or(&0);
                if curr_view <= view {
                    if verify_confirm(consensus_message, &self.committee) {
                        self.views.insert(*slot, *view);
                        return true;
                    }
                }
                return false;

                // Ensure that the QC is valid, and that we are in the same view
                //qc.verify(&self.committee).is_ok() && self.views.get(slot).unwrap() == view
                //self.views.get(slot).unwrap() == view && verify_confirm(consensus_message, &self.committee)
            }
            ConsensusMessage::Commit {
                slot,
                view,
                qc,
                proposals,
                aggregate_report: _,
            } => {
                verify_commit(consensus_message, &self.committee)

                // Ensure that the QC is valid, and that we are in the same view
                //qc.verify(&self.committee).is_ok() && self.views.get(slot).unwrap() == view
                //self.views.get(slot).unwrap() == view && verify_commit(consensus_message, &self.committee)
            }
        }
    }

    async fn is_consensus_ready(&mut self, header: &Header) -> bool {
        let mut is_ready = true;
        for (_, consensus_message) in &header.consensus_messages {
            match consensus_message {
                ConsensusMessage::Prepare {
                    slot: _,
                    view: _,
                    tc: _,
                    qc_ticket: _,
                    proposals: _,
                    aggregate_report: _,
                } => {
                    // Consensus is ready if all proposals for all prepare messages in a car aren't
                    // missing
                    // NOTE: If view > 0 then don't have to call this, only the leader of first
                    // view takes responsibility, only check for view > 0 so you don't need to
                    // check whether winning proposals is correct
                    // TODO: For testing with faults make the certificate of tip syncing happen
                    // asynchronously, change synchronizer so that it write to the store without
                    // the history
                    is_ready = is_ready
                        && !self
                            .synchronizer
                            .get_proposals(consensus_message, header)
                            .await
                            .unwrap()
                            .is_empty();
                }
                ConsensusMessage::Commit {
                    slot: _,
                    view: _,
                    qc: _,
                    proposals: _,
                    aggregate_report: _,
                } => {
                    //TODO: If we'd like to process it earlier
                    //self.process_commit_message(consensus_message.clone(), &Header::default()).await?;
                }
                _ => {}
            };
        }
        is_ready
    }

    #[async_recursion]
    async fn process_consensus_messages(
        &mut self,
        header: &Header,
    ) -> DagResult<Vec<(Slot, Digest, Signature)>> {
        // Map between consensus instance digest and a signature indicating a vote for that
        // instance
        let mut consensus_votes: Vec<(Slot, Digest, Signature)> = Vec::new();

        for (_, consensus_message) in &header.consensus_messages {
            //println!("processing instance");
            debug!("processing instance");
            if self.is_valid(consensus_message).await {
                match consensus_message {
                    ConsensusMessage::Prepare {
                        slot,
                        view: _,
                        tc: _,
                        qc_ticket: _,
                        proposals,
                        aggregate_report: _,
                    } => {
                        //println!("processing prepare message");
                        debug!(
                            "processing prepare in slot {:?} with proposal {:?}",
                            slot, proposals
                        );
                        self.process_prepare_message(consensus_message, consensus_votes.as_mut())
                            .await;
                    }
                    ConsensusMessage::Confirm {
                        slot,
                        view: _,
                        qc: _,
                        proposals,
                        aggregate_report: _,
                    } => {
                        //println!("processing confirm message");
                        debug!(
                            "processing confirm in slot {:?} with proposal {:?}",
                            slot, proposals
                        );
                        // Start syncing on the proposals if we haven't already
                        self.synchronizer
                            .get_proposals(consensus_message, &header)
                            .await?;
                        self.process_confirm_message(consensus_message, consensus_votes.as_mut())
                            .await;
                    }
                    ConsensusMessage::Commit {
                        slot,
                        view: _,
                        qc: _,
                        proposals: _,
                        aggregate_report: _,
                    } => {
                        //println!("processing commit message");
                        debug!("processing commit in slot {:?}", slot);
                        self.process_commit_message(consensus_message.clone(), &header)
                            .await?; //FIXME: Does this need to be a copy?
                    }
                }
            }
        }

        //println!("Returning from process consensus size of consensus sigs {:?}", consensus_votes.len());
        Ok(consensus_votes)
    }

    async fn process_consensus_request(
        &mut self,
        consensus_req: ConsensusRequest,
    ) -> DagResult<()> {
        let consensus_message = &consensus_req.message;
        debug!("received consensus request for slot");

        match consensus_message {
            ConsensusMessage::Prepare {
                slot,
                view: _,
                tc: _,
                qc_ticket: _,
                proposals,
                aggregate_report: _,
            } => {
                debug!(
                    "processing prepare in slot {:?} with proposal {:?}",
                    slot, proposals
                );
            }
            ConsensusMessage::Confirm {
                slot,
                view: _,
                qc: _,
                proposals,
                aggregate_report: _,
            } => {
                debug!(
                    "processing confirm in slot {:?} with proposal {:?}",
                    slot, proposals
                );
            }
            ConsensusMessage::Commit {
                slot,
                view: _,
                qc: _,
                proposals: _,
                aggregate_report: _,
            } => {
                debug!("processing commit in slot {:?}", slot);
            }
        }
        let dig = consensus_message.digest();
        match &consensus_message {
            //TODO: Re-factor ConsensusMessages to all have slot/view, option for TC/QC, and a type.
            ConsensusMessage::Prepare {
                slot,
                view,
                tc: _,
                qc_ticket: _,
                proposals: _,
                aggregate_report: _,
            } => {
                self.consensus_instances
                    .insert((*slot, dig.clone()), consensus_message.clone());
            }
            ConsensusMessage::Confirm {
                slot,
                view,
                qc: _,
                proposals: _,
                aggregate_report: _,
            } => {
                self.consensus_instances
                    .insert((*slot, dig.clone()), consensus_message.clone());
            }
            _ => {}
        };

        debug!("try to verify");
        let mut valid = true;
        if consensus_req.author != self.name {
            consensus_req.verify(&self.committee)?;
            debug!("check validity");
        }

        valid = self.is_valid(&consensus_message).await;

        if !valid {
            debug!("not valid");
            return Ok(());
        }

        self.process_consensus_message(consensus_req.message, consensus_req.author)
            .await
    }

    async fn process_consensus_message(
        &mut self,
        consensus_message: ConsensusMessage,
        author: PublicKey,
    ) -> DagResult<()> {
        let mut consensus_votes: Vec<(Slot, Digest, Signature)> = Vec::new();

        debug!("processing consensus msg");

        let mut header = Header::default();
        header.author = author;

        match &consensus_message {
            ConsensusMessage::Prepare {
                slot,
                view: _,
                tc: _,
                qc_ticket: _,
                proposals,
                aggregate_report: _,
            } => {
                debug!(
                    "processing prepare in slot {:?} with proposal {:?}",
                    slot, proposals
                );

                // Optimistic tips not ready, reschedule for processing
                if self.use_optimistic_tips
                    && !self
                        .synchronizer
                        .optimistic_tips_ready(&consensus_message, &header)
                        .await?
                {
                    debug!("optimistic tips not ready");
                    return Ok(());
                } else {
                    if !self.use_optimistic_tips {
                        self.synchronizer
                            .get_proposals(&consensus_message, &header)
                            .await?;
                        debug!("start syncing certified proposals");
                    } else {
                        debug!("optimistic tips are ready");
                    }
                }

                // Start syncing on the proposals if we haven't already
                //self.synchronizer.get_proposals(&consensus_message, &header).await?;
                self.process_prepare_message(&consensus_message, consensus_votes.as_mut())
                    .await;
            }
            ConsensusMessage::Confirm {
                slot,
                view: _,
                qc: _,
                proposals,
                aggregate_report: _,
            } => {
                //println!("processing confirm message");
                debug!(
                    "processing confirm in slot {:?} with proposal {:?}",
                    slot, proposals
                );
                // Start syncing on the proposals if we haven't already
                self.synchronizer
                    .get_proposals(&consensus_message, &header)
                    .await?;
                self.process_confirm_message(&consensus_message, consensus_votes.as_mut())
                    .await;
            }
            ConsensusMessage::Commit {
                slot,
                view: _,
                qc: _,
                proposals: _,
                aggregate_report: _,
            } => {
                //println!("processing commit message");
                debug!("processing commit in slot {:?}", slot);
                self.process_commit_message(consensus_message.clone(), &header)
                    .await?; //FIXME: Does this need to be a copy?
            }
        }

        debug!(
            "Returning from process consensus size of consensus sigs {:?}",
            consensus_votes.len()
        );

        //TODO: Check that process_prepare isn't doing more than necessary for external.
        if consensus_votes.is_empty() {
            //E.g. if it was a Commit, or if messages were invalid.
            return Ok(());
        }

        //Broadcast the vote.
        let (slot, digest, sig) = consensus_votes.pop().unwrap();
        let vote = ConsensusVote {
            author: self.name,
            slot,
            digest,
            sig,
        };

        if author == self.name {
            debug!("Process own consensus vote");
            self.process_consensus_vote(vote, false)
                .await
                .expect("Failed to process our own vote"); //TODO: Don't need to sign...
        } else {
            debug!("Send consensus vote to replica {}", author);

            self.send_msg(
                PrimaryMessage::ConsensusVote(vote),
                slot,
                Some(author),
                true,
            )
            .await;
        }

        Ok(())
    }

    //#[async_recursion]
    async fn process_prepare_message(
        &mut self,
        prepare_message: &ConsensusMessage,
        consensus_sigs: &mut Vec<(Slot, Digest, Signature)>,
    ) {
        match prepare_message {
            ConsensusMessage::Prepare {
                slot,
                view,
                tc: _,
                qc_ticket,
                proposals,
                aggregate_report,
            } => {
                // Check if this prepare message can be used for a ticket to propose in the next slot
                // TODO: Remove from process_header
                let x = self.is_prepare_ticket_ready(prepare_message).await;

                // View-1 timeout for this slot starts when its Prepare is processed.
                if *view == 1
                    && !self.committed_slots.contains_key(slot)
                    && !self.timers.contains(&(*slot, 1))
                {
                    debug!("start timer for slot {} view 1", slot);
                    self.timer_futures
                        .push(Box::pin(Timer::new(*slot, 1, self.timeout_delay)));
                    self.timers.insert((*slot, 1));
                }

                for (pk, proposal) in proposals {
                    debug!(
                        "prepare slot {:?}, validator: {:?}, proposal height {:?}",
                        slot, pk, proposal.height
                    );

                    // Log prepare event for lane vector construction for each validator
                    if let Some(mut logger_guard) = get_metrics_logger() {
                        if let Some(logger) = logger_guard.as_mut() {
                            let prepare_details = serde_json::json!({
                                "validator_pk": hex::encode(pk.0),
                                "proposal_height": proposal.height,
                                "slot": slot
                            });
                            logger.log_event("prepare", prepare_details);
                        }
                    }
                }
                debug!(
                    "during simulated partition is {:?} for slot {:?}",
                    self.during_simulated_asynchrony, slot
                );
                debug!("prepare vote in slot {:?}", slot);

                // Ensure that we don't vote for another prepare in this slot, view
                self.last_voted_consensus.insert((*slot, *view));

                if self.use_fast_path {
                    // Already checked that we were in the right view from validity checks, so just insert into our local high_proposals map
                    self.high_proposals.insert(
                        *slot,
                        ConsensusMessage::Prepare {
                            slot: *slot,
                            view: *view,
                            tc: None,
                            qc_ticket: None,
                            proposals: proposals.clone(),
                            aggregate_report: aggregate_report.clone(),
                        },
                    ); //Note: Don't need to store TC or QC's.
                }

                // Indicate that we vote for this instance's prepare message
                //let sig = Signature::default();
                let sig = self
                    .signature_service
                    .request_signature(prepare_message.digest())
                    .await;
                consensus_sigs.push((*slot, prepare_message.digest(), sig));
                debug!(
                    "Prepare-Vote for slot: {}, view: {},has digest: {}",
                    slot,
                    view,
                    prepare_message.digest()
                );
            }
            _ => {}
        }
    }

    //#[async_recursion]
    async fn process_confirm_message(
        &mut self,
        confirm_message: &ConsensusMessage,
        consensus_sigs: &mut Vec<(Slot, Digest, Signature)>,
    ) {
        match confirm_message {
            ConsensusMessage::Confirm {
                slot,
                view,
                qc,
                proposals: _,
                aggregate_report: _,
            } => {
                // Already checked that we were in the right view from validity checks, so just
                // insert into our local high_qc map
                self.high_qcs.insert(*slot, confirm_message.clone());

                // Indicate that we vote for this instance's confirm message
                //let sig = Signature::default();
                let sig = self
                    .signature_service
                    .request_signature(confirm_message.digest())
                    .await;
                consensus_sigs.push((*slot, confirm_message.digest(), sig));
                debug!(
                    "Confirm-Vote for slot: {}, view: {}, qc_dig {:?} -> has digest: {}",
                    slot,
                    view,
                    qc.id,
                    confirm_message.digest()
                );
            }
            _ => {}
        }
    }

    fn enough_coverage(
        &mut self,
        prepare_proposals: &HashMap<PublicKey, Proposal>,
        //current_proposals: &HashMap<PublicKey, Proposal>,
    ) -> bool {
        let current_proposals = match self.use_optimistic_tips {
            true => &self.current_proposal_tips,
            false => &self.current_certified_tips,
        };

        // Checks whether there have been n-f new certs from the proposals from the ticket
        let new_tips: HashMap<&PublicKey, &Proposal> = current_proposals
            .iter()
            .filter(|(pk, proposal)| proposal.height > prepare_proposals.get(&pk).unwrap().height)
            .collect();

        debug!("current proposals {:?}", current_proposals);
        debug!("prepare proposal tips {:?}", prepare_proposals);

        debug!("Cut condition = {}", self.cut_condition_type);
        return new_tips.len() as u8 >= self.cut_condition_type;
    }

    #[async_recursion]
    async fn process_commit_message(
        &mut self,
        commit_message: ConsensusMessage,
        header: &Header,
    ) -> DagResult<()> {
        debug!("Called process commit");
        match &commit_message {
            ConsensusMessage::Commit {
                slot,
                view,
                qc,
                proposals,
                aggregate_report,
            } => {
                debug!("Try to commit slot {}", slot);
                // Start simulating async once slot 1 is committed
                if self.simulate_asynchrony && *slot == 1 && !self.already_set_timers {
                    debug!("added async timers");

                    self.already_set_timers = true;
                    debug!("asynchrony start is {:?}", self.asynchrony_start);
                    for i in 0..self.asynchrony_start.len() {
                        // Determine whether this node is targeted for this async window.
                        // If explicit node ids are provided for this window, use them; otherwise fall back to first-N logic.
                        let explicit_targets = self
                            .asynchrony_node_ids_per_window
                            .get(i)
                            .cloned()
                            .unwrap_or_default();
                        let is_targeted = if !explicit_targets.is_empty() {
                            explicit_targets.contains(&self.node_index)
                        } else {
                            let mut keys: Vec<_> =
                                self.committee.authorities.keys().cloned().collect();
                            keys.sort();
                            let index = keys.binary_search(&self.name).unwrap_or(0);
                            index < self.affected_nodes[i] as usize
                        };

                        if self.asynchrony_type[i] == AsyncEffectType::Failure {
                            // Skip nodes that are not affected by the asynchrony
                            if !is_targeted {
                                continue;
                            }
                        }

                        if self.asynchrony_type[i] == AsyncEffectType::Egress
                            || self.asynchrony_type[i] == AsyncEffectType::PrepareDelay
                        {
                            // Skip nodes that are not affected by the asynchrony
                            if !is_targeted {
                                continue;
                            }
                        }

                        if self.asynchrony_type[i] == AsyncEffectType::VoteDelay {
                            let mut keys: Vec<_> =
                                self.committee.authorities.keys().cloned().collect();
                            keys.sort();
                            let index = keys.binary_search(&self.name).unwrap_or(0);
                            info!(
                                "vote delay: node_index={} pk_index={} targeted={} explicit_targets={:?}",
                                self.node_index, index, is_targeted, explicit_targets
                            );
                            // Skip nodes that are not affected by the asynchrony
                            if !is_targeted {
                                info!("vote delay skipping node {:?}", self.name);
                                continue;
                            } else {
                                info!("vote delay not skipping node {:?}", self.name);
                            }
                        }

                        let start_offset = self.asynchrony_start[i].saturating_mul(1000);
                        let duration_ms = self.asynchrony_duration[i].saturating_mul(1000);
                        let end_offset = start_offset + duration_ms;

                        let async_start = Timer::new(0, 0, start_offset);
                        let async_end = Timer::new(0, 0, end_offset);

                        self.async_timer_futures.push(Box::pin(async_start));
                        self.async_timer_futures.push(Box::pin(async_end));

                        if self.asynchrony_type[i] == AsyncEffectType::Partition
                            || self.current_effect_type == AsyncEffectType::Equivocate
                        {
                            let mut keys: Vec<_> =
                                self.committee.authorities.keys().cloned().collect();
                            keys.sort();
                            let index = keys.binary_search(&self.name).unwrap();

                            // Figure out which partition we are in, partition_nodes indicates when the left partition ends
                            let mut start: usize = 0;
                            let mut end: usize = 0;

                            // We are in the right partition
                            if index > self.affected_nodes[i] as usize - 1 {
                                start = self.affected_nodes[i] as usize;
                                end = keys.len();
                            } else {
                                // We are in the left partition
                                start = 0;
                                end = self.affected_nodes[i] as usize;
                            }

                            // These are the nodes in our side of the partition
                            for j in start..end {
                                self.partition_public_keys.insert(keys[j]);
                            }

                            debug!("partition pks are {:?}", self.partition_public_keys);
                        }
                    }
                }

                // Stop all timers for this slot across views.
                self.timers.retain(|(s, _)| *s != *slot);
                self.high_qcs.insert(*slot, commit_message.clone());

                let sl = *slot;
                //update bounding heuristic
                self.last_committed_slot = max(sl, self.last_committed_slot);
                if let Some(transition_end) = self.k_transition_end_slot {
                    if self.last_committed_slot >= transition_end {
                        info!(
                            "✅ k transition completed at slot {} (active k={})",
                            transition_end, self.k
                        );
                        self.k_transition_end_slot = None;
                    }
                }
                // Drain-based pending k activation (safer than fixed transition_end).
                self.try_activate_pending_k();
                let first_commit_observation = self
                    .committed_slots
                    .insert(
                        sl,
                        CommitQC::new(
                            *slot,
                            *view,
                            qc.clone(),
                            proposals.clone(),
                            aggregate_report.clone(),
                        )
                        .await,
                    )
                    .is_none();
                if first_commit_observation {
                    let event_type = if qc.votes.len() == self.committee.size() {
                        "fast_path"
                    } else {
                        "slow_path"
                    };
                    if let Some(mut logger_guard) = get_metrics_logger() {
                        if let Some(logger) = logger_guard.as_mut() {
                            // logger.log_event(
                            //     "commit",
                            //     serde_json::json!({
                            //         "slot": *slot,
                            //         "view": *view,
                            //         "qc_votes": qc.votes.len(),
                            //         "path_type": event_type,
                            //         "message": format!("commit observed for slot {:?} via {}", slot, event_type)
                            //     }),
                            // );
                            logger.log_event(
                                event_type,
                                serde_json::json!({
                                    "slot": *slot,
                                    "view": *view,
                                    "qc_votes": qc.votes.len(),
                                    "path_type": event_type,
                                    "message": format!("commit observed for slot {:?} via {}", slot, event_type)
                                }),
                            );
                        }
                    }
                }
                self.persist_coordination_certificate(*slot, *view, qc)
                    .await?;

                // Apply pending parameter update when reaching applied_begin in that epoch.
                // If that window is already past, drop the update instead of rolling it forward.
                if let Some(target_epoch) = self.pending_param_update_epoch {
                    if self.applied_begin == 0 || self.epoch_slots == 0 {
                        self.apply_pending_parameters(target_epoch).await;
                    } else if let Some(pos_in_epoch) = self.epoch_slot_index(*slot) {
                        let current_epoch = self.epoch_index_for_slot(*slot);
                        // First commit at or past applied_begin in the target epoch.
                        // Exact equality misses k-parallel out-of-order commits.
                        if pos_in_epoch >= self.applied_begin && current_epoch == target_epoch
                        {
                            info!(
                                "✅ Applying parameter update after slot {} committed (epoch {}, pos={})",
                                *slot, target_epoch, pos_in_epoch
                            );
                            self.apply_pending_parameters(target_epoch).await;
                        } else if current_epoch > target_epoch {
                            self.drop_param_update(target_epoch, *slot);
                        }
                    }
                }

                // Check if we need to trigger metrics collection based on transaction count
                // NEW: Trigger when total_tx % epoch_transactions == window_transactions
                // This ensures metrics are collected based on actual work (tx count) rather than slot count
                // if self.epoch_transactions > 0 && self.window_transactions > 0 {
                //     let current_epoch = self.total_committed_transactions / self.epoch_transactions;
                //     let position_in_epoch = self.total_committed_transactions % self.epoch_transactions;

                //     // Trigger when we've just passed the window_transactions mark in this epoch
                //     if position_in_epoch >= self.window_transactions
                //         && self.total_committed_transactions > self.last_triggered_tx_count
                //         && self.last_triggered_tx_count < current_epoch * self.epoch_transactions + self.window_transactions {

                //         self.last_triggered_tx_count = self.total_committed_transactions;

                //         info!("📊 Triggering metrics collection: total_txs={}, epoch={}, epoch_txs={}, window_txs={}, slot={}",
                //                self.total_committed_transactions, current_epoch, self.epoch_transactions,
                //                self.window_transactions, self.last_committed_slot);

                //         // Send metrics request asynchronously (no waiting for response)
                //         if let Err(e) = self.send_metrics_request(self.last_committed_slot).await {
                //             warn!("Failed to send metrics request at tx_count {}: {}", self.total_committed_transactions, e);
                //         }
                //     }
                // }

                // Slot-based trigger: slots start from 1, each epoch covers exactly window_size slots.
                // Epoch k (0-based): [k*window_size+1, (k+1)*window_size+1) i.e. slots k*W+1 .. (k+1)*W
                // Trigger when committed_slot is a positive multiple of window_size (last slot of epoch).
                if self.window_size > 0 {
                    let committed_slot = self.last_committed_slot;
                    let current_epoch = committed_slot / self.epoch_slots;
                    let pos_in_epoch = committed_slot % self.epoch_slots;
                    let last_epoch = if self.last_triggered_slot == 0 {
                        None
                    } else {
                        Some(self.last_triggered_slot / self.epoch_slots)
                    };

                    // Trigger once per epoch when we've reached (or passed) window_size.
                    if pos_in_epoch >= self.window_size
                        && last_epoch.map_or(true, |e| current_epoch > e)
                    {
                        self.last_triggered_slot = committed_slot;

                        info!(
                            "📊 Triggering metrics collection: slot={}, epoch_slots={}, window_size={}",
                            committed_slot, self.epoch_slots, self.window_size
                        );

                        if let Err(e) = self.send_metrics_request(committed_slot).await {
                            warn!(
                                "Failed to send metrics request for slot {}: {}",
                                committed_slot, e
                            );
                        }
                    }
                }

                // Check if we need to update parameters at epoch boundary
                // Trigger when next_slot % epoch_slots == 0
                // if self.epoch_slots > 0 {
                //     let next_slot = self.last_committed_slot + 1;
                //     if next_slot % self.epoch_slots == 0 {
                //         info!("🔄 Epoch boundary reached (slot {}), checking for parameter updates...", self.last_committed_slot);

                //         // Check and update parameters asynchronously (no waiting for response)
                //         let _ = self.check_and_update_parameters().await;
                //     }
                // }

                let next_slot_k = self.effective_k_for_qc_slot(*slot);
                let timer_slot = *slot + next_slot_k;
                if next_slot_k == 1
                    && !self.committed_slots.contains_key(&timer_slot)
                    && !self.timers.contains(&(timer_slot, 1))
                {
                    debug!("start timer for slot {} view 1", timer_slot);
                    self.timer_futures
                        .push(Box::pin(Timer::new(timer_slot, 1, self.timeout_delay)));
                    self.timers.insert((timer_slot, 1));
                }
                // k>1: do not start slot+k's view-1 timer here. That slot may still
                // be waiting for cut-condition coverage. try_prepare_waiting_slots
                // below starts the timer once the ticket is actually actionable.

                // // Only send to committer if proposals and all ancestors are stored locally,
                // // otherwise sync will be triggered, and this commit message will be reprocessed
                if !self
                    .synchronizer
                    .get_proposals(&commit_message, &header)
                    .await
                    .unwrap()
                    .is_empty()
                {
                    //println!("Sent to committer");
                    debug!("sending to committer");
                    // Only write to the log if we aren't the failed node during simulated asynchrony
                    let write_to_log = !(self.during_simulated_asynchrony
                        && self.current_effect_type == AsyncEffectType::Failure);
                    debug!("write to log is {}", write_to_log);

                    self.tx_committer
                        .send((commit_message.clone(), write_to_log))
                        .await
                        .expect("Failed to send headers");
                }

                // // Count transactions in this commit for metrics collection (inlined to avoid Send issues with async_recursion)
                // let mut tx_count_in_commit = 0u64;
                // for (_, proposal) in proposals {
                //     // Read the header from store to get batch metadata
                //     if let Ok(Some(bytes)) = self.store.read(proposal.header_digest.to_vec()).await {
                //         if let Ok(header) = bincode::deserialize::<Header>(&bytes) {
                //             // Count transactions in all batches of this header
                //             for (digest, _worker_id) in &header.payload {
                //                 if let Some(metadata) = header.batch_metadata.get(digest) {
                //                     tx_count_in_commit += metadata.transaction_count as u64;
                //                 }
                //             }
                //         }
                //     }
                // }

                // Update slot mapping for current epoch window
                // Logic:
                // - slot_start (epoch_start): First slot that commits to a new epoch (after crossing k*epoch_transactions boundary)
                // - slot_end (epoch_window): First slot that crosses window_transactions threshold within this epoch
                // Collection window is [slot_start, slot_end], representing [epoch_start, epoch_start+window_transactions)
                // if self.epoch_transactions > 0 && self.window_transactions > 0 && tx_count_in_commit > 0 {
                //     let position_in_epoch = self.total_committed_transactions % self.epoch_transactions;
                //     let new_position = (self.total_committed_transactions + tx_count_in_commit) % self.epoch_transactions;
                //     let current_epoch = self.total_committed_transactions / self.epoch_transactions;
                //     let new_epoch = (self.total_committed_transactions + tx_count_in_commit) / self.epoch_transactions;

                //     // Record slot_start (epoch_start): First slot that commits transactions to a new epoch
                //     // Only record when:
                //     // 1. slot_start is not yet set (is_none), AND
                //     // 2. We're crossing an epoch boundary (new_epoch > current_epoch) OR it's the very first commit
                //     if self.epoch_slot_start.is_none() {
                //         if new_epoch > current_epoch || self.total_committed_transactions == 0 {
                //             self.epoch_slot_start = Some(sl);
                //             debug!("📍 Epoch slot_start recorded: slot={}, total_tx={}, epoch={}->{}, position={}->{}",
                //                    sl, self.total_committed_transactions, current_epoch, new_epoch, position_in_epoch, new_position);
                //         }
                //     }

                //     // Record slot_end (epoch_window): First slot that crosses window_transactions threshold in this epoch
                //     // This marks the end of the collection window
                //     if position_in_epoch < self.window_transactions && new_position >= self.window_transactions {
                //         self.epoch_slot_end = Some(sl);
                //         debug!("📍 Epoch slot_end recorded: slot={}, total_tx={}->{}, position={}->{}",
                //                sl, self.total_committed_transactions, self.total_committed_transactions + tx_count_in_commit,
                //                position_in_epoch, new_position);
                //     }
                // }

                // self.total_committed_transactions += tx_count_in_commit;
                // debug!("Committed {} transactions, total: {}", tx_count_in_commit, self.total_committed_transactions);

                // }

                //Try waking any prepares that are waiting for a QC ticket
                self.try_prepare_waiting_slots().await?;

                // Garbage collect (can be ascyn)
                //self.clean_slot(sl);
                self.clean_slot_periods(sl);
            }
            _ => {}
        }

        Ok(())
    }

    #[async_recursion]
    async fn clean_slot(&mut self, slot: Slot) -> DagResult<()> {
        //GC Consensus instances
        self.consensus_instances.retain(|(s, _), _| s != &slot);
        self.consensus_cancel_handlers.retain(|s, _| s != &slot);

        //GC QC_Makers
        self.qc_makers.retain(|(s, _), _| s != &slot);
        // self.pqc_makers.retain(|(s, _), _| s != &sl);
        // self.cqc_makers.retain(|(s, _), _| s != &sl);
        Ok(())
    }

    #[async_recursion]
    async fn clean_slot_periods(&mut self, slot: Slot) -> DagResult<()> {
        //slot periodics
        let k = self.effective_k_for_qc_slot(slot).max(1);
        let slot_period = slot % k;

        //GC Consensus instances
        self.consensus_instances
            .retain(|(s, _), _| s % k != slot_period && s <= &slot);
        self.consensus_cancel_handlers
            .retain(|s, _| s % k != slot_period && s <= &slot);
        //self.committed_slots GC those that are older.

        //GC QC_Makers
        self.qc_makers
            .retain(|(s, _), _| s % k != slot_period && s <= &slot);

        Ok(())
    }

    #[async_recursion]
    async fn process_loopback(
        &mut self,
        consensus_message: ConsensusMessage,
        header: Header,
    ) -> DagResult<()> {
        //println!("reprocessing a header/commit message");
        debug!(
            "Can reprocess a header/commit message for header {:?}, consensus message {:?}",
            header, consensus_message
        );
        match &consensus_message {
            ConsensusMessage::Prepare {
                slot,
                view,
                tc: _,
                qc_ticket: _,
                proposals,
                aggregate_report: _,
            } => {
                if self.use_ride_share {
                    // Now that proposals are ready we can reprocess the header
                    self.process_header(header, false).await?;
                } else {
                    if self.last_voted_consensus.contains(&(*slot, *view)) {
                        //Don't prepare twice
                        return Ok(());
                    }

                    self.process_consensus_message(consensus_message, header.author)
                        .await?
                }
            }
            ConsensusMessage::Confirm {
                slot: _,
                view: _,
                qc: _,
                proposals: _,
                aggregate_report: _,
            } => {
                // Don't need to do anything for the confirm case, since proposals will be
                // sent to the committer once a commit message is received
            }
            ConsensusMessage::Commit {
                slot: _,
                view: _,
                qc: _,
                proposals: _,
                aggregate_report: _,
            } => {
                // Send the commit message to the committer to order everything
                let write_to_log = !(self.during_simulated_asynchrony
                    && self.current_effect_type == AsyncEffectType::Failure);
                debug!("write to log is {}", write_to_log);
                self.tx_committer
                    .send((consensus_message, write_to_log))
                    .await
                    .expect("Failed to send to committer");
            }
        };
        Ok(())
    }

    async fn process_forwarded_message(
        &mut self,
        consensus_message: ConsensusMessage,
    ) -> DagResult<()> {
        match &consensus_message {
            ConsensusMessage::Prepare {
                slot: _,
                view: _,
                tc: _,
                qc_ticket: _,
                proposals: _,
                aggregate_report: _,
            } => {
                // We have a ticket for instance (slot + 1, 1), so check if we have enough coverage
                // to send a prepare message, otherwise buffer it
                self.is_prepare_ticket_ready(&consensus_message).await?;
            }
            ConsensusMessage::Commit {
                slot: _,
                view: _,
                qc: _,
                proposals: _,
                aggregate_report: _,
            } => {
                // Process any forwarded commit messages
                // NOTE: Used "dummy header" for second argument for now, header doesn't matter since proposal syncing
                // does not block processing the header, only prepare messages do
                self.process_commit_message(consensus_message, &self.current_header.clone())
                    .await?;
            }
            _ => {}
        }
        Ok(())
    }

    fn calculate_timeout(&self, slot: Slot, view: View) -> u64 {
        // Don't use exponential timeouts for the first few slots since nodes are just booting up
        if slot > 4 && self.use_expoential_timeouts {
            let timeout = self.timeout_delay as f64 * 2.0_f64.powi((view - 1) as i32);
            debug!("Timeout for view {} is {}", view, timeout as u64);
            timeout as u64
        } else {
            debug!("Timeout for view {} is {}", view, self.timeout_delay);
            self.timeout_delay
        }
    }

    async fn qc_timeout() {}

    async fn local_timeout_round(&mut self, slot: Slot, view: View) -> DagResult<()> {
        warn!(
            "Timeout reached for slot {}, view {}. Leader is {}",
            slot,
            view,
            self.leader_elector.get_leader(slot, view)
        );
        //println!("timeout was triggered");

        //If timer was cancelled, ignore  -- Note: technically redundant with commit check below, but currently we do not insert CommitQC's... TODO: Need to insert these so we can avoid joining view change and just reply.
        if !self.timers.contains(&(slot, view)) {
            debug!(
                "Timer for slot {}, view {} is obsolete. Has been cancelled",
                slot, view
            );
            return Ok(());
        }

        // If timing out a smaller view than the current view, ignore
        match self.views.get(&slot) {
            Some(v) => {
                if *v > view {
                    debug!(
                        "Timer for slot {}, view {} is obsolete. Have moved to view {}",
                        slot, view, *v
                    );
                    return Ok(());
                }
            }
            None => {}
        };

        // If we have already committed then ignore the timeout.
        if self.committed_slots.contains_key(&slot) {
            debug!(
                "Timer for slot {}, view {} is obsolete. Slot already committed",
                slot, view
            );
            return Ok(());
        }

        debug!("Sending Timeout for slot {}, view {}", slot, view);
        // Make a timeout message.for the slot, view, containing the highest QC this replica has
        // seen
        let timeout = Timeout::new(
            slot,
            view,
            self.high_qcs.get(&slot).cloned(),
            self.high_proposals.get(&slot).cloned(),
            self.name,
            self.signature_service.clone(),
        )
        .await;
        debug!("Created Timeout: {:?}", timeout);

        // Broadcast the timeout message.
        debug!("Broadcasting Timeout: {:?}", timeout);
        /*let addresses = self
            .committee
            .others_primaries(&self.name)
            .iter()
            .map(|(_, x)| x.primary_to_primary)
            .collect();
        let message = bincode::serialize(&PrimaryMessage::Timeout(timeout.clone()))
            .expect("Failed to serialize timeout message");
        let handlers = self.network
            .broadcast(addresses, Bytes::from(message))
            .await;

        self.consensus_cancel_handlers
            .entry(slot)
            .or_insert_with(Vec::new)
            .extend(handlers);*/

        self.send_msg(PrimaryMessage::Timeout(timeout.clone()), slot, None, true)
            .await;

        //println!("Processed our own timeout");
        // Process our message.
        self.handle_timeout(&timeout).await?;

        // Liveness: Timeout futures are single-shot. If a TC was not assembled, the
        // slot would otherwise stall forever with no further timer. Re-arm until we
        // either commit or advance the view via TC.
        if !self.committed_slots.contains_key(&slot) {
            let curr_view = self.views.get(&slot).copied().unwrap_or(0);
            if curr_view <= view {
                let duration = self.calculate_timeout(slot, view);
                self.timer_futures
                    .push(Box::pin(Timer::new(slot, view, duration)));
                self.timers.insert((slot, view));
                debug!(
                    "Re-armed timeout for slot {}, view {} (duration={}ms)",
                    slot, view, duration
                );
            }
        }

        Ok(())
    }

    async fn handle_timeout(&mut self, timeout: &Timeout) -> DagResult<()> {
        debug!(
            "Processing timeout {:?} for slot {}, view {}",
            timeout, timeout.slot, timeout.view
        );

        // TODO: If already committed then don't need to verify, just forward commit

        // Don't process timeout messages for old views
        match self.views.get(&timeout.slot) {
            Some(view) => {
                if timeout.view < *view {
                    return Ok(());
                }
            }
            _ => {}
        };

        debug!("past timeout old view check");

        if self.committed_slots.contains_key(&timeout.slot) {
            //TODO: Forward CommitQC instead.
            return Ok(());
        }

        debug!("past timeout commit check");

        // Ensure the timeout is well formed.
        timeout.verify(&self.committee)?;

        debug!("past timeout verify check");

        // If we haven't seen a timeout for this slot, view, then create a new TC maker for it.
        if self.tc_makers.get(&(timeout.slot, timeout.view)).is_none() {
            self.tc_makers
                .insert((timeout.slot, timeout.view), TCMaker::new());
        }

        // Otherwise, get the TC maker for this slot, view.
        let tc_maker = self
            .tc_makers
            .get_mut(&(timeout.slot, timeout.view))
            .unwrap();

        //println!("got tc maker");
        debug!("got tc maker");

        // Add the new vote to our aggregator and see if we have a quorum.
        if let Some(tc) = tc_maker.append(timeout.clone(), &self.committee)? {
            debug!("Assembled TimeoutCertificate {:?}", tc);

            // Try to advance the view
            self.views.insert(timeout.slot, timeout.view + 1);

            // Start the new view timer
            let duration = self.calculate_timeout(tc.slot, tc.view + 1);
            let timer = Timer::new(tc.slot, tc.view + 1, duration);
            self.timer_futures.push(Box::pin(timer));
            self.timers.insert((tc.slot, tc.view + 1));

            // Broadcast the TC.
            // TODO: Low priority: If you see f+1 timeouts then join the mutiny

            //FIXME: Don't need to broadcast TC if we join mutiny upon seeing f+1 timeouts.
            // debug!("Broadcasting {:?}", tc);
            // let addresses = self
            //     .committee
            //     .others_primaries(&self.name)
            //     .iter()
            //     .map(|(_, x)| x.primary_to_primary)
            //     .collect();
            // let message = bincode::serialize(&PrimaryMessage::TC(tc.clone()))
            //     .expect("Failed to serialize timeout certificate");
            // let handlers = self.network
            //     .broadcast(addresses, Bytes::from(message))
            //     .await;

            // self.consensus_cancel_handlers
            //     .entry(slot)
            //     .or_insert_with(Vec::new)
            //     .extend(handlers);

            // Generate a new prepare if we are the next leader.
            self.generate_prepare_from_tc(&tc).await?;
        }
        //println!("return from handle timeout");
        Ok(())
    }

    async fn generate_prepare_from_tc(&mut self, tc: &TC) -> DagResult<()> {
        // Make a new prepare message if we are the next leader.
        if self.name == self.leader_elector.get_leader(tc.slot, tc.view + 1) {
            debug!("IsLeader. Start prepare from TC");
            let winning_proposals = tc.get_winning_proposals(&self.committee);

            debug!("winning proposals: {:?}", winning_proposals);

            // If there is no QC we have to propose, then use our current tips for our proposal => happens later
            // if winning_proposals.is_empty() {
            //     winning_proposals = self.current_proposal_tips.clone();
            // }

            // Create a prepare message for the next view, containing the ticket and proposals
            // TODO: Low priority can make winning proposals empty
            let expected_epoch = self.epoch_index_for_slot(tc.slot);
            let aggregate_report = self.get_pending_aggregate_report(expected_epoch);
            if let Some(report) = aggregate_report.as_ref() {
                info!(
                    "📦 Embedding AggregateReport digest {} into Prepare(slot={}, view={}) from TC",
                    report.digest(),
                    tc.slot,
                    tc.view + 1
                );
            }

            let prepare_message: ConsensusMessage = ConsensusMessage::Prepare {
                slot: tc.slot,
                view: tc.view + 1,
                tc: Some(tc.clone()),
                qc_ticket: None,
                proposals: winning_proposals.clone(),
                aggregate_report,
            };
            if self.use_ride_share {
                self.tx_info
                    .send(prepare_message.clone())
                    .await
                    .expect("Failed to send consensus instance");
            } else {
                self.send_consensus_req(prepare_message).await?;
            }
        }
        Ok(())
    }

    async fn handle_tc(&mut self, tc: &TC) -> DagResult<()> {
        debug!("Processing TC {:?}", tc);
        // Generate a new prepare if we are the next leader.
        self.generate_prepare_from_tc(tc).await?;

        Ok(())
    }

    fn sanitize_header(&mut self, header: &Header) -> DagResult<()> {
        /*ensure!(
            self.gc_round <= header.height,
            DagError::HeaderTooOld(header.id.clone(), header.height)
        );*/

        // Verify the header's signature.
        header.verify(&self.committee)?;

        // TODO [issue #3]: Prevent bad nodes from sending junk headers with high round numbers.

        Ok(())
    }

    fn sanitize_vote(&mut self, vote: &Vote) -> DagResult<()> {
        //Check:
        //If vote has no consensus sigs and vote.aggregator already has QC => ignore vote.
        if self.current_header.id.eq(&vote.id) && self.votes_aggregator.complete {
            if vote.consensus_votes.is_empty() {
                //Note: If vote is empty, but self.current_header.consensus_messages is not we can still ignore processing this vote (since it requires no consensus processing)
                return Err(DagError::CarAlreadySatisfied);
            } else {
                //Don't need to check signature (won't use it), but do need to process vote for consensus contents
                return Ok(());
            }
        }

        // Verify the vote.
        vote.verify(&self.committee).map_err(DagError::from)
    }

    fn sanitize_certificate(&mut self, certificate: &Certificate) -> DagResult<()> {
        ensure!(
            self.gc_round <= certificate.height(),
            DagError::CertificateTooOld(certificate.digest(), certificate.height())
        );

        //println!("Past first ensure");

        // Verify the certificate (and the embedded header).
        certificate.verify(&self.committee).map_err(DagError::from)
    }

    pub async fn send_msg(
        &mut self,
        message: PrimaryMessage,
        height: u64,
        author: Option<PublicKey>,
        consensus_handler: bool,
    ) {
        // Fallback to original logic when no simulator
        match self.current_effect_type {
            AsyncEffectType::Off => {
                debug!("message sent normally");
                self.send_msg_normal(message, height, author, consensus_handler)
                    .await;
            }
            AsyncEffectType::TempBlip => {
                // Keep original logic as fallback
                match message {
                    PrimaryMessage::ConsensusMessage(m) => match m.clone() {
                        ConsensusMessage::Prepare {
                            slot,
                            view,
                            tc,
                            qc_ticket: _,
                            proposals,
                            aggregate_report: _,
                        } => {
                            debug!("Simulating Asynchrony: skip sending Prepare for slot {} view {}. This will trigger a view change", slot, view);
                            self.async_delayed_prepare = Some(m);
                        }
                        _ => {}
                    },
                    _ => {
                        debug!("send all other messages")
                    }
                }
                panic!("TempBlip currently deprecated");
            }

            // AsyncEffectType::Failure => {
            //     match message.clone() {
            //         PrimaryMessage::ConsensusMessage(m) => {
            //             match m.clone() {
            //                 ConsensusMessage::Prepare {slot, view, tc, qc_ticket: _, proposals} => {
            //                     self.async_delayed_prepare = Some(m);
            //                     if self.dropped_slot > 0 {
            //                         self.send_msg_normal(message, height, author, consensus_handler).await;
            //                     } else {
            //                         self.dropped_slot = slot;
            //                     }
            //                 },
            //                 ConsensusMessage::Confirm { slot, view: _, qc: _, proposals: _ } => {
            //                     if self.dropped_slot > 0 {
            //                         self.send_msg_normal(message, height, author, consensus_handler).await;
            //                     } else {
            //                         self.dropped_slot = slot;
            //                     }
            //                 },
            //                 ConsensusMessage::Commit { slot, view: _, qc: _, proposals: _ } => {
            //                     if self.dropped_slot > 0 {
            //                         self.send_msg_normal(message, height, author, consensus_handler).await;
            //                     } else {
            //                         self.dropped_slot = slot;
            //                     }
            //                 }
            //             }
            //         }
            //         _ => { debug!("dropping all other messages") }
            //     }
            //     debug!("dropping message");
            // }
            AsyncEffectType::Failure => {
                match message.clone() {
                    PrimaryMessage::ConsensusMessage(m) => match m.clone() {
                        ConsensusMessage::Prepare {
                            slot,
                            view,
                            tc,
                            qc_ticket: _,
                            proposals,
                            aggregate_report: _,
                        } => {
                            debug!("dropping Prepare for slot {} view {}", slot, view);
                        }
                        ConsensusMessage::Confirm {
                            slot,
                            view,
                            qc: _,
                            proposals: _,
                            aggregate_report: _,
                        } => {
                            debug!("dropping Confirm for slot {} view {}", slot, view);
                        }
                        ConsensusMessage::Commit {
                            slot,
                            view,
                            qc: _,
                            proposals: _,
                            aggregate_report: _,
                        } => {
                            debug!("dropping Commit for slot {} view {}", slot, view);
                        }
                    },
                    _ => {
                        self.send_msg_normal(message, height, author, consensus_handler)
                            .await;
                    }
                }
                debug!("dropping message");
            }

            AsyncEffectType::Partition => match author {
                Some(author) => {
                    if self.partition_public_keys.contains(&author) {
                        debug!("single message during partition, sent normally");
                        self.send_msg_normal(message, height, Some(author), consensus_handler)
                            .await;
                    } else {
                        debug!("single message during partition, buffered");
                        self.partition_delayed_msgs.push((
                            message,
                            height,
                            Some(author),
                            consensus_handler,
                        ));
                    }
                }
                None => {
                    if self.partition_public_keys.len() > 1 {
                        self.send_msg_partition(&message, height, consensus_handler, true)
                            .await;
                        debug!(
                            "broadcast message during partition, sent to nodes in our partition"
                        );
                    }
                    self.partition_delayed_msgs
                        .push((message, height, None, consensus_handler));
                }
            },
            AsyncEffectType::Equivocate => {
                match message.clone() {
                    PrimaryMessage::Header(curr_header, sync_flag) => {
                        // Only malicious nodes (first f+1 nodes) perform equivocation
                        if !self.is_malicious_node() {
                            debug!("Equivocate: honest node, sending normally");
                            self.send_msg_normal(
                                PrimaryMessage::Header(curr_header, sync_flag),
                                height,
                                author,
                                consensus_handler,
                            )
                            .await;
                            return;
                        }

                        // Malicious node: send different headers to different partitions
                        let fake_header_partition_0 =
                            self.create_fake_header(&curr_header, 0).await;
                        let fake_header_partition_1 =
                            self.create_fake_header(&curr_header, 1).await;

                        debug!("Malicious node equivocate: curr_header_id={:?}, fake_0_id={:?}, fake_1_id={:?}", 
                               curr_header.id, fake_header_partition_0.id, fake_header_partition_1.id);

                        match author {
                            Some(pk) => {
                                // Singlecast: determine which partition the target belongs to
                                let mut keys: Vec<_> =
                                    self.committee.authorities.keys().cloned().collect();
                                keys.sort();
                                let target_index = keys.binary_search(&pk).unwrap_or(0);
                                let partition_cut =
                                    self.affected_nodes.front().cloned().unwrap_or(0) as usize;

                                if target_index < partition_cut {
                                    debug!("Equivocate singlecast: sending normal header to pk={:?} (partition 0)", pk);
                                    self.send_msg_normal(
                                        PrimaryMessage::Header(curr_header, sync_flag),
                                        height,
                                        Some(pk),
                                        consensus_handler,
                                    )
                                    .await;
                                } else {
                                    debug!("Equivocate singlecast: sending FAKE_1 header to pk={:?} (partition 1)", pk);
                                    self.send_msg_normal(
                                        PrimaryMessage::Header(fake_header_partition_1, sync_flag),
                                        height,
                                        Some(pk),
                                        consensus_handler,
                                    )
                                    .await;
                                }
                            }
                            None => {
                                // Broadcast: send different headers to different partitions
                                if self.partition_public_keys.len() > 0 {
                                    debug!("Malicious node equivocate broadcast: sending different headers to partitions");
                                    // Send fake_header_0 to partition 0 (our partition)
                                    let msg_fake_0 = PrimaryMessage::Header(curr_header, sync_flag);
                                    self.send_msg_partition(
                                        &msg_fake_0,
                                        height,
                                        consensus_handler,
                                        true,
                                    )
                                    .await;
                                    // Send fake_header_1 to partition 1 (other partition)
                                    let msg_fake_1 =
                                        PrimaryMessage::Header(fake_header_partition_1, sync_flag);
                                    self.send_msg_partition(
                                        &msg_fake_1,
                                        height,
                                        consensus_handler,
                                        false,
                                    )
                                    .await;
                                } else {
                                    debug!("Equivocate: no partition_public_keys set, sending normally");
                                    self.send_msg_normal(
                                        PrimaryMessage::Header(curr_header, sync_flag),
                                        height,
                                        None,
                                        consensus_handler,
                                    )
                                    .await;
                                }
                            }
                        }
                    }
                    _ => {
                        // Non-header messages go through normally
                        self.send_msg_normal(message, height, author, consensus_handler)
                            .await;
                    }
                }
            }
            AsyncEffectType::Egress => {
                let egress_end_time = Instant::now() + Duration::from_millis(self.egress_penalty);
                debug!("current time is {:?}", Instant::now());
                debug!("egress penalty is {:?}", self.egress_penalty);
                debug!("msg egress end time is {:?}", egress_end_time);
                let actual_send_time = egress_end_time.min(self.current_egress_end);
                debug!("msg actual send time is {:?}", actual_send_time);
                self.egress_delay_queue.insert_at(
                    (message, height, author, consensus_handler),
                    actual_send_time,
                );
            }

            AsyncEffectType::PrepareDelay => match message.clone() {
                PrimaryMessage::ConsensusMessage(m) => match m.clone() {
                    ConsensusMessage::Prepare {
                        slot,
                        view,
                        tc: _,
                        qc_ticket: _,
                        proposals: _,
                        aggregate_report: _,
                    } => {
                        debug!(
                            "Simulating Prepare Delay: delay Prepare for slot {} view {}",
                            slot, view
                        );
                        let egress_end_time =
                            Instant::now() + Duration::from_millis(self.egress_penalty);
                        debug!("current time is {:?}", Instant::now());
                        debug!("egress penalty is {:?}", self.egress_penalty);
                        debug!("msg egress end time is {:?}", egress_end_time);
                        let actual_send_time = egress_end_time.min(self.current_egress_end);
                        debug!("msg actual send time is {:?}", actual_send_time);
                        self.egress_delay_queue.insert_at(
                            (message, height, author, consensus_handler),
                            actual_send_time,
                        );
                    }
                    _ => {}
                },
                PrimaryMessage::ConsensusRequest(m) => match m.message.clone() {
                    ConsensusMessage::Prepare {
                        slot,
                        view,
                        tc: _,
                        qc_ticket: _,
                        proposals: _,
                        aggregate_report: _,
                    } => {
                        debug!(
                            "Simulating Prepare Delay: delay Prepare for slot {} view {}",
                            slot, view
                        );
                        let egress_end_time =
                            Instant::now() + Duration::from_millis(self.egress_penalty);
                        debug!("current time is {:?}", Instant::now());
                        debug!("egress penalty is {:?}", self.egress_penalty);
                        debug!("msg egress end time is {:?}", egress_end_time);
                        let actual_send_time = egress_end_time.min(self.current_egress_end);
                        debug!("msg actual send time is {:?}", actual_send_time);
                        self.egress_delay_queue.insert_at(
                            (message, height, author, consensus_handler),
                            actual_send_time,
                        );
                    }
                    _ => {}
                },
                _ => {
                    debug!("Simulating Prepare Delay: sending message normally");
                    self.send_msg_normal(message, height, author, consensus_handler)
                        .await;
                }
            },

            AsyncEffectType::VoteDelay => {
                if let PrimaryMessage::ConsensusVote(_) = &message {
                    info!("Simulating Vote Delay: delay Vote");
                    let egress_end_time =
                        Instant::now() + Duration::from_millis(self.egress_penalty);
                    info!("current time is {:?}", Instant::now());
                    info!("egress penalty is {:?}", self.egress_penalty);
                    info!("msg egress end time is {:?}", egress_end_time);
                    self.egress_delay_queue.insert_at(
                        (message, height, author, consensus_handler),
                        egress_end_time,
                    );
                } else {
                    info!("Simulating Vote Delay: sending message normally");
                    self.send_msg_normal(message, height, author, consensus_handler)
                        .await;
                }
                return;
            }

            _ => {
                panic!("not a valid effect")
            }
        }
    }

    pub async fn send_msg_partition(
        &mut self,
        message: &PrimaryMessage,
        height: u64,
        consensus_handler: bool,
        our_partition: bool,
    ) {
        let addresses = self
            .committee
            .others_primaries(&self.name)
            .iter()
            .filter(|(pk, _)| {
                (our_partition && self.partition_public_keys.contains(pk))
                    || (!our_partition && !self.partition_public_keys.contains(pk))
            })
            .map(|(_, x)| x.primary_to_primary)
            .collect();
        debug!(
            "addresses for partition are are {:?}, our partition is {}",
            addresses, our_partition
        );

        let bytes = bincode::serialize(message).expect("Failed to serialize message");
        let handlers = self.network.broadcast(addresses, Bytes::from(bytes)).await;
        if consensus_handler {
            self.consensus_cancel_handlers
                .entry(height)
                .or_insert_with(Vec::new)
                .extend(handlers);
        } else {
            self.cancel_handlers
                .entry(height)
                .or_insert_with(Vec::new)
                .extend(handlers);
        }
    }

    pub async fn send_msg_normal(
        &mut self,
        message: PrimaryMessage,
        height: u64,
        author: Option<PublicKey>,
        consensus_handler: bool,
    ) {
        match author {
            Some(author) => {
                let address = self
                    .committee
                    .primary(&author)
                    .expect("Author of valid header is not in the committee")
                    .primary_to_primary;
                debug!(
                    "send_msg_normal: UNICAST -> target={:?}, addr={:?}, msg={:?}, consensus_handler={}",
                    author,
                    address,
                    message,
                    consensus_handler
                );
                let bytes = bincode::serialize(&message).expect("Failed to serialize message");
                let handler = self.network.send(address, Bytes::from(bytes)).await;
                if consensus_handler {
                    self.consensus_cancel_handlers
                        .entry(height)
                        .or_insert_with(Vec::new)
                        .push(handler);
                } else {
                    self.cancel_handlers
                        .entry(height)
                        .or_insert_with(Vec::new)
                        .push(handler);
                }
            }
            None => {
                let targets: Vec<PublicKey> = self
                    .committee
                    .others_primaries(&self.name)
                    .iter()
                    .map(|(pk, _)| pk.clone())
                    .collect();
                let addresses = self
                    .committee
                    .others_primaries(&self.name)
                    .iter()
                    .map(|(_, x)| x.primary_to_primary)
                    .collect();
                debug!(
                    "send_msg_normal: BROADCAST -> num_targets={}, targets={:?}, msg={:?}, consensus_handler={}",
                    targets.len(),
                    targets,
                    message,
                    consensus_handler
                );

                let bytes = bincode::serialize(&message).expect("Failed to serialize message");
                let handlers = self.network.broadcast(addresses, Bytes::from(bytes)).await;
                if consensus_handler {
                    self.consensus_cancel_handlers
                        .entry(height)
                        .or_insert_with(Vec::new)
                        .extend(handlers);
                } else {
                    self.cancel_handlers
                        .entry(height)
                        .or_insert_with(Vec::new)
                        .extend(handlers);
                }
            }
        }
    }

    // Check for parameter updates from .parameters.json file by compsaring current values with file values
    fn read_parameters_file(&self) -> Option<serde_json::Value> {
        let parameters_path = ".parameters.json";
        debug!("parameters path is {:?}", parameters_path);
        match fs::read_to_string(&parameters_path) {
            Ok(content) => match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(json_value) => Some(json_value),
                Err(e) => {
                    warn!("Failed to parse parameters.json: {}", e);
                    None
                }
            },
            Err(e) => {
                warn!("Failed to read parameters.json: {}", e);
                None
            }
        }
    }

    async fn apply_parameter_updates(&mut self, json_value: &serde_json::Value) {
        let mut parameters_updated = false;

        // Check and update parameters (excluding the ones user specified not to update)
        // if let Some(timeout_delay) = json_value.get("timeout_delay").and_then(|v| v.as_u64()) {
        //     if self.timeout_delay != timeout_delay {
        //         self.timeout_delay = timeout_delay;
        //         info!("✅ Updated timeout_delay: {} -> {}", self.timeout_delay, timeout_delay);
        //         parameters_updated = true;
        //     }
        // }
        if let Some(header_size) = json_value.get("header_size").and_then(|v| v.as_u64()) {
            // Note: header_size is used by proposer, not stored in Core struct
            if self.header_size != header_size as usize {
                info!(
                    "✅ Updated header_size: {} -> {}",
                    self.header_size, header_size
                );
                self.header_size = header_size as usize;
                parameters_updated = true;
            }
        }
        if let Some(max_header_delay) = json_value.get("max_header_delay").and_then(|v| v.as_u64())
        {
            // Note: max_header_delay is used by proposer, not stored in Core struct
            if self.max_header_delay != max_header_delay {
                info!(
                    "✅ Updated max_header_delay: {} -> {}",
                    self.max_header_delay, max_header_delay
                );
                self.max_header_delay = max_header_delay;
                parameters_updated = true;
            }
        }
        if let Some(use_optimistic_tips) = json_value
            .get("use_optimistic_tips")
            .and_then(|v| v.as_bool())
        {
            if self.use_optimistic_tips != use_optimistic_tips {
                info!(
                    "✅ Updated use_optimistic_tips: {} -> {}",
                    self.use_optimistic_tips, use_optimistic_tips
                );
                self.use_optimistic_tips = use_optimistic_tips;
                parameters_updated = true;
            }
        }

        // if let Some(sync_retry_delay) = json_value.get("sync_retry_delay").and_then(|v| v.as_u64()) {
        //     // Note: sync_retry_delay is not stored in Core struct, might need to be added if needed
        //     info!("ℹ️  sync_retry_delay parameter exists in file but not applied (not stored in Core)");
        // }
        // if let Some(sync_retry_nodes) = json_value.get("sync_retry_nodes").and_then(|v| v.as_u64()) {
        //     // Note: sync_retry_nodes parameter exists in file but not applied (not stored in Core)
        //     info!("ℹ️  sync_retry_nodes parameter exists in file but not applied (not stored in Core)");
        // }
        if let Some(batch_size) = json_value.get("batch_size").and_then(|v| v.as_u64()) {
            // Note: batch_size is used by workers, not stored in Core struct
            if self.batch_size != batch_size as usize {
                info!(
                    "✅ Updated batch_size: {} -> {}",
                    self.batch_size, batch_size
                );
                self.batch_size = batch_size as usize;
                parameters_updated = true;
            }
        }
        if let Some(max_batch_delay) = json_value.get("max_batch_delay").and_then(|v| v.as_u64()) {
            // Note: max_batch_delay is used by workers, not stored in Core struct
            if self.max_batch_delay != max_batch_delay {
                info!(
                    "✅ Updated max_batch_delay: {} -> {}",
                    self.max_batch_delay, max_batch_delay
                );
                self.max_batch_delay = max_batch_delay;
                parameters_updated = true;
            }
        }
        if let Some(use_optimistic_tips) = json_value
            .get("use_optimistic_tips")
            .and_then(|v| v.as_bool())
        {
            if self.use_optimistic_tips != use_optimistic_tips {
                info!(
                    "✅ Updated use_optimistic_tips: {} -> {}",
                    self.use_optimistic_tips, use_optimistic_tips
                );
                self.use_optimistic_tips = use_optimistic_tips;
                parameters_updated = true;
            }
        }
        if let Some(k) = json_value.get("k").and_then(|v| v.as_u64()) {
            if self.k != k || self.pending_k.map(|pk| pk != k).unwrap_or(false) {
                let previous_k = self.k;
                if k < previous_k {
                    // Safer decrease: keep self.k at the old value for in-flight ticket
                    // validation, but restrict NEW opens to pending_k immediately.
                    // Activate self.k = pending_k only once open instances <= pending_k.
                    self.pending_k = Some(k);
                    self.k_transition_end_slot = None;
                    info!(
                        "🛡️  k decrease pending: {} -> {} (open_slots={}); restricting new opens immediately",
                        previous_k,
                        k,
                        self.open_instance_count()
                    );
                    self.try_activate_pending_k();
                } else {
                    // Increase (or replace a pending decrease with a larger target): apply now.
                    if let Some(pk) = self.pending_k.take() {
                        info!(
                            "⏩ Clearing pending k={} because new target k={} is not a decrease",
                            pk, k
                        );
                    }
                    info!("✅ Updated k (parallel_proposals): {} -> {}", previous_k, k);
                    self.last_used_k = previous_k;
                    self.k = k;
                    self.k_transition_end_slot = None;
                }
                info!(
                    "ℹ️  k transition state: last_used_k={}, active_k={}, pending_k={:?}, transition_end={:?}",
                    self.last_used_k, self.k, self.pending_k, self.k_transition_end_slot
                );
                parameters_updated = true;
            }
        }

        if let Some(fast_path_timeout) =
            json_value.get("fast_path_timeout").and_then(|v| v.as_u64())
        {
            if self.fast_path_timeout != fast_path_timeout {
                info!(
                    "✅ Updated fast_path_timeout: {} -> {}",
                    self.fast_path_timeout, fast_path_timeout
                );
                self.fast_path_timeout = fast_path_timeout;
                if fast_path_timeout == 0 {
                    info!("🔄 Fast path timeout is 0, setting use_fast_path to false");
                    self.use_fast_path = false;
                } else {
                    info!("🔄 Fast path timeout is not 0, setting use_fast_path to true");
                    self.use_fast_path = true;
                }
                parameters_updated = true;
            }
        }
        // if let Some(use_ride_share) = json_value.get("use_ride_share").and_then(|v| v.as_bool()) {
        //     if self.use_ride_share != use_ride_share {
        //         self.use_ride_share = use_ride_share;
        //         info!("✅ Updated use_ride_share: {} -> {}", self.use_ride_share, use_ride_share);
        //         parameters_updated = true;
        //     }
        // }
        // if let Some(car_timeout) = json_value.get("car_timeout").and_then(|v| v.as_u64()) {
        //     if self.car_timeout != car_timeout {
        //         self.car_timeout = car_timeout;
        //         info!("✅ Updated car_timeout: {} -> {}", self.car_timeout, car_timeout);
        //         parameters_updated = true;
        //     }
        // }
        if let Some(cut_condition_type) = json_value
            .get("cut_condition_type")
            .and_then(|v| v.as_u64())
        {
            let cut_condition_u8 = cut_condition_type as u8;
            if self.cut_condition_type != cut_condition_u8 {
                self.cut_condition_type = cut_condition_u8;
                info!(
                    "✅ Updated cut_condition_type: {} -> {}",
                    self.cut_condition_type, cut_condition_u8
                );
                parameters_updated = true;
            }
        }

        // Note: use_exponential_timeouts is not updated per user request
        // Note: epoch_slots is not updated per user request
        // Note: window_size is not updated per user request
        // Note: affected_nodes and asynchrony related fields are not updated per user request

        // Send parameter updates to proposer
        if let Some(header_size) = json_value.get("header_size").and_then(|v| v.as_u64()) {
            if let Some(max_header_delay) =
                json_value.get("max_header_delay").and_then(|v| v.as_u64())
            {
                // Send header_size and max_header_delay to proposer
                if let Err(e) = self
                    .tx_proposer_params
                    .send((header_size as usize, max_header_delay))
                    .await
                {
                    warn!("Failed to send parameter updates to proposer: {}", e);
                } else {
                    info!("📤 Sent parameter updates to proposer: header_size={}, max_header_delay={}", header_size, max_header_delay);
                }
            }
        }

        // Create parameter update signal file to notify workers
        match std::fs::File::create(PARAMETER_UPDATE_SIGNAL_FILE) {
            Ok(_) => info!("🚩 Created parameter update signal file to notify all workers"),
            Err(e) => warn!("Failed to create parameter update signal file: {}", e),
        }

        if parameters_updated {
            info!(
                "🔄 Parameters have been updated from .parameters-{}",
                self.node_index
            );
        } else {
            debug!("No parameter changes detected in .parameters.json");
        }
    }

    fn capture_pending_parameters(&mut self) {
        if self.pending_param_update_params.is_some() {
            return;
        }
        if let Some(json_value) = self.read_parameters_file() {
            self.pending_param_update_params = Some(json_value);
        } else {
            warn!("Failed to capture pending parameters from file");
        }
    }

    async fn apply_pending_parameters(&mut self, target_epoch: u64) {
        if let Some(json_value) = self
            .pending_param_update_params
            .take()
            .or_else(|| self.read_parameters_file())
        {
            self.apply_parameter_updates(&json_value).await;
            self.param_apply_ok = true;
            self.param_apply_ok_epoch = target_epoch;
            info!("📌 param_apply_ok=true for epoch {}", target_epoch);
        } else {
            warn!("Pending parameter update has no captured values");
            self.param_apply_ok = false;
            self.param_apply_ok_epoch = target_epoch;
        }
        self.pending_param_update_epoch = None;
        self.rl_param_signal_epoch = self.rl_param_signal_epoch.saturating_add(1);
    }

    fn drop_param_update(&mut self, missed_epoch: u64, last_slot: u64) {
        self.param_apply_ok = false;
        self.param_apply_ok_epoch = missed_epoch;
        self.pending_param_update_epoch = None;
        self.pending_param_update_params = None;
        self.rl_param_signal_epoch = missed_epoch.saturating_add(1);
        info!(
            "🗑️  Dropping late parameter update for epoch {} (slot {})",
            missed_epoch, last_slot
        );
    }

    fn inject_param_apply_ok(&self, state_json: String, epoch: u64) -> String {
        let mut root = match serde_json::from_str::<serde_json::Value>(&state_json) {
            Ok(value) => value,
            Err(_) => return state_json,
        };
        // Report whether the previous epoch's action landed at its applied_begin.
        // Epoch 0 has no predecessor, so the first report is always true.
        let measured_epoch = epoch.saturating_sub(1);
        let apply_ok = epoch == 0
            || (self.param_apply_ok && self.param_apply_ok_epoch == measured_epoch);
        if let Some(object) = root.as_object_mut() {
            object.insert("param_apply_ok".to_string(), serde_json::json!(apply_ok));
        }
        serde_json::to_string(&root).unwrap_or(state_json)
    }

    async fn check_and_update_parameters(&mut self) {
        if let Some(json_value) = self.read_parameters_file() {
            self.apply_parameter_updates(&json_value).await;
        }
    }

    async fn handle_param_update_signal(&mut self, message: String) -> DagResult<()> {
        let signal_epoch = match serde_json::from_str::<serde_json::Value>(&message)
            .ok()
            .and_then(|v| v.get("epoch").and_then(|e| e.as_u64()))
        {
            Some(epoch) => epoch,
            None => {
                warn!("Invalid parameter update signal payload, falling back to expected epoch");
                self.rl_param_signal_epoch
            }
        };

        info!(
            "🚩 RL parameter update signal received via socket (epoch {})",
            signal_epoch
        );

        let last_slot = self.last_committed_slot;
        let current_epoch = self.epoch_index_for_slot(last_slot);

        if self.epoch_slots == 0 || self.applied_begin == 0 {
            let _ = self.check_and_update_parameters().await;
            self.param_apply_ok = true;
            self.param_apply_ok_epoch = signal_epoch;
            self.rl_param_signal_epoch = signal_epoch.saturating_add(1);
            return Ok(());
        }

        if signal_epoch < self.rl_param_signal_epoch {
            info!(
                "⏭️  Skipping stale parameter signal for epoch {} (expected {})",
                signal_epoch, self.rl_param_signal_epoch
            );
            return Ok(());
        }

        if let Some(pending_epoch) = self.pending_param_update_epoch {
            info!(
                "⏭️  Skipping parameter signal for epoch {} (pending epoch {} still waiting for applied_begin)",
                signal_epoch, pending_epoch
            );
            return Ok(());
        }

        if signal_epoch < current_epoch {
            // Late relative to the signaled epoch: still install now so the
            // replica does not keep stale parameters. Credit apply_ok to the
            // signaled epoch so the dump that measures it can report success
            // if it has not been written yet.
            info!(
                "⚠️  Late parameter signal for epoch {} (current epoch {}), applying immediately",
                signal_epoch, current_epoch
            );
            self.pending_param_update_epoch = Some(signal_epoch);
            self.capture_pending_parameters();
            self.apply_pending_parameters(signal_epoch).await;
            self.rl_param_signal_epoch = current_epoch.saturating_add(1);
            return Ok(());
        }

        if signal_epoch > current_epoch {
            self.rl_param_signal_epoch = signal_epoch;
            self.pending_param_update_epoch = Some(signal_epoch);
            self.capture_pending_parameters();
            info!(
                "⏳ Parameter update pending for future epoch {} (current epoch {}, last slot {})",
                signal_epoch, current_epoch, last_slot
            );
            return Ok(());
        }

        if let Some(pos_in_epoch) = self.epoch_slot_index(last_slot) {
            if pos_in_epoch >= self.applied_begin {
                self.pending_param_update_epoch = Some(signal_epoch);
                self.capture_pending_parameters();
                info!(
                    "✅ Parameter update applying immediately: pos {} >= applied_begin {} in epoch {}",
                    pos_in_epoch, self.applied_begin, signal_epoch
                );
                self.apply_pending_parameters(signal_epoch).await;
            } else {
                self.pending_param_update_epoch = Some(signal_epoch);
                self.capture_pending_parameters();
                info!(
                    "⏳ Parameter update pending: will apply after slot {} in epoch {}, current slot is {}",
                    self.applied_begin, signal_epoch, last_slot
                );
            }
        }

        Ok(())
    }

    fn epoch_index_for_slot(&self, slot: u64) -> u64 {
        if self.epoch_slots == 0 || slot == 0 {
            0
        } else {
            (slot.saturating_sub(1)) / self.epoch_slots
        }
    }

    fn epoch_slot_index(&self, slot: u64) -> Option<u64> {
        if self.epoch_slots == 0 || slot == 0 {
            None
        } else {
            Some((slot.saturating_sub(1) % self.epoch_slots) + 1)
        }
    }

    fn epoch_start_slot(&self, epoch: u64) -> Option<u64> {
        if self.epoch_slots == 0 {
            None
        } else {
            Some(epoch.saturating_mul(self.epoch_slots).saturating_add(1))
        }
    }

    fn base_k_for_slot(&self, slot: u64) -> u64 {
        if self.epoch_slots == 0 || self.applied_begin == 0 {
            return self.k;
        }
        match self.epoch_slot_index(slot) {
            Some(pos_in_epoch) if pos_in_epoch < self.applied_begin => self.last_used_k,
            Some(_) => self.k,
            None => self.k,
        }
    }

    fn k_with_transition_guard(&self, slot: u64, base_k: u64) -> u64 {
        match self.k_transition_end_slot {
            Some(transition_end) if slot <= transition_end && self.last_used_k > base_k => {
                self.last_used_k
            }
            _ => base_k,
        }
    }

    fn slot_already_started(&self, slot: Slot) -> bool {
        self.already_proposed_slots.contains(&slot)
            || self.committed_slots.contains_key(&slot)
            || self.views.contains_key(&slot)
            || self.timers.iter().any(|(s, _)| *s == slot)
    }

    fn open_instance_count(&self) -> u64 {
        self.already_proposed_slots
            .iter()
            .filter(|slot| **slot > 0 && !self.committed_slots.contains_key(slot))
            .count() as u64
    }

    fn max_allowed_ticket_k(&self) -> u64 {
        // During pending decrease, self.k remains the old (larger) value so in-flight
        // tickets with the old distance remain acceptable.
        self.k
            .max(self.last_used_k)
            .max(self.pending_k.unwrap_or(0))
            .max(1)
    }

    fn try_activate_pending_k(&mut self) {
        let Some(pending) = self.pending_k else {
            return;
        };
        let open = self.open_instance_count();
        if open > pending {
            debug!(
                "⏳ Pending k={} not ready yet (open_slots={})",
                pending, open
            );
            return;
        }
        let previous = self.k;
        self.last_used_k = previous;
        self.k = pending;
        self.pending_k = None;
        self.k_transition_end_slot = None;
        info!(
            "✅ Activated pending k decrease: {} -> {} (open_slots={})",
            previous, pending, open
        );
    }

    fn effective_k_for_prepare_slot(&self, prepare_slot: u64) -> u64 {
        // While a decrease is pending, newly opened slots must use the target (smaller) k
        // immediately; already-started slots keep the old k for ticket distance checks.
        if let Some(pending) = self.pending_k {
            if self.slot_already_started(prepare_slot) {
                return self.k.max(1);
            }
            return pending.max(1);
        }
        let base_k = self.base_k_for_slot(prepare_slot);
        self.k_with_transition_guard(prepare_slot, base_k).max(1)
    }

    fn effective_k_for_qc_slot(&self, qc_slot: u64) -> u64 {
        // Prefer implied ticket distance in validation paths. This helper remains for
        // callers that still estimate k from a QC slot during/after transitions.
        if self.pending_k.is_some() {
            return self.k.max(1);
        }
        let base_k = self.base_k_for_slot(qc_slot);
        self.k_with_transition_guard(qc_slot, base_k).max(1)
    }

    async fn handle_metrics_state_message(&mut self, message: String) -> DagResult<()> {
        let response: serde_json::Value = match serde_json::from_str(&message) {
            Ok(value) => value,
            Err(e) => {
                warn!("Invalid metrics response JSON: {}", e);
                return Ok(());
            }
        };

        let status = response
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if status != "collected" {
            warn!("Metrics response not collected: {}", message);
            return Ok(());
        }

        let epoch = match response.get("epoch").and_then(|v| v.as_u64()) {
            Some(epoch) => epoch,
            None => {
                warn!("Metrics response missing epoch: {}", message);
                return Ok(());
            }
        };

        let state_file = match response.get("state_file").and_then(|v| v.as_str()) {
            Some(path) => path.to_string(),
            None => {
                warn!("Metrics response missing state_file: {}", message);
                return Ok(());
            }
        };

        let state_json = match tokio::fs::read_to_string(&state_file).await {
            Ok(contents) => contents,
            Err(e) => {
                warn!("Failed to read state file {}: {}", state_file, e);
                return Ok(());
            }
        };

        // ── Data-pollution ablation ──────────────────────────────────────────
        // If this node is in the polluter set and the RNG rolls below the
        // configured probability, rewrite key metric fields before signing and
        // broadcasting the StateReport. The local copy is always the true one.
        let state_json = if self.data_pollution_node_ids.contains(&self.node_index)
            && self.data_pollution_prob > 0.0
            && rand::thread_rng().gen::<f64>() < self.data_pollution_prob
        {
            info!(
                "Polluting state json for epoch {} with strategy {:?}",
                epoch, self.data_pollution_strategy
            );
            Self::pollute_state_json(state_json, &self.data_pollution_strategy)
        } else {
            state_json
        };
        let state_json = self.inject_param_apply_ok(state_json, epoch);

        let report =
            StateReport::new(epoch, state_json, self.name, &mut self.signature_service).await;
        self.record_state_report(report.clone());

        match serde_json::from_str::<serde_json::Value>(&report.state) {
            Ok(state_value) => {
                info!("StateReport state(local): {}", state_value);
                let reward_value = state_value.get("end_to_end_metrics");
                info!(
                    "StateReport reward(end_to_end_metrics, local): {}",
                    reward_value.unwrap_or(&serde_json::Value::Null)
                );
            }
            Err(e) => {
                warn!("Failed to parse local StateReport state JSON: {}", e);
                info!("StateReport state(raw, local): {}", report.state);
            }
        }

        let height = self.last_committed_slot;
        self.send_msg_normal(PrimaryMessage::StateReport(report), height, None, false)
            .await;

        info!("📣 Broadcast StateReport for epoch {}", epoch);
        self.ensure_collection_timer(epoch);
        self.try_build_aggregate_report(epoch, false);

        Ok(())
    }

    async fn handle_state_report(&mut self, report: StateReport) -> DagResult<()> {
        if let Err(e) = report.verify(&self.committee) {
            warn!(
                "Invalid StateReport from {} (epoch {}): {}",
                report.author, report.epoch, e
            );
            return Ok(());
        }

        let inserted = self.record_state_report(report.clone());
        if inserted {
            info!(
                "📥 Received StateReport from {} for epoch {}",
                report.author, report.epoch
            );
            match serde_json::from_str::<serde_json::Value>(&report.state) {
                Ok(state_value) => {
                    info!("StateReport state: {}", state_value);
                    let reward_value = state_value.get("end_to_end_metrics");
                    info!(
                        "StateReport reward(end_to_end_metrics): {}",
                        reward_value.unwrap_or(&serde_json::Value::Null)
                    );
                }
                Err(e) => {
                    warn!("Failed to parse StateReport state JSON: {}", e);
                    info!("StateReport state(raw): {}", report.state);
                }
            }
            self.ensure_collection_timer(report.epoch);
            self.try_build_aggregate_report(report.epoch, false);
        }
        Ok(())
    }

    fn record_state_report(&mut self, report: StateReport) -> bool {
        let epoch_reports = self
            .state_reports
            .entry(report.epoch)
            .or_insert_with(HashMap::new);
        epoch_reports.insert(report.author, report).is_none()
    }

    fn ensure_collection_timer(&mut self, epoch: u64) {
        if self.active_collection_timers.contains(&epoch) {
            return;
        }
        self.active_collection_timers.insert(epoch);
        let timeout_ms = self.collection_timeout_ms;
        let fut = async move {
            tokio::time::sleep(Duration::from_millis(timeout_ms)).await;
            epoch
        };
        self.collection_timer_futures.push(Box::pin(fut));
    }

    fn try_build_aggregate_report(&mut self, epoch: u64, allow_fallback: bool) {
        if self.aggregated_epochs.contains(&epoch) {
            return;
        }
        let Some(reports) = self.state_reports.get(&epoch) else {
            return;
        };
        let report_count = reports.len();
        let f = (self.committee.size().saturating_sub(1)) / 3;
        let threshold = if allow_fallback { 2 * f + 1 } else { 3 * f + 1 };
        if report_count < threshold {
            return;
        }

        let reports_vec: Vec<StateReport> = reports.values().cloned().collect();
        let aggregate = AggregateReport::new(epoch, reports_vec);
        self.pending_aggregate_reports
            .entry(epoch)
            .or_insert(aggregate);
        self.aggregated_epochs.insert(epoch);
        info!(
            "✅ AggregateReport ready for epoch {} (reports={}, threshold={}, fallback={})",
            epoch, report_count, threshold, allow_fallback
        );
    }

    fn get_pending_aggregate_report(&self, expected_epoch: u64) -> Option<AggregateReport> {
        self.pending_aggregate_reports.get(&expected_epoch).cloned()
    }

    async fn persist_coordination_certificate(
        &mut self,
        slot: Slot,
        view: View,
        qc: &QC,
    ) -> DagResult<()> {
        let Some(report) = self
            .certifiable_aggregate_reports
            .get(&(slot, view))
            .cloned()
        else {
            return Ok(());
        };
        let epoch = report.epoch;
        let digest = report.digest();

        if self.certified_aggregate_epochs.contains(&epoch) {
            return Ok(());
        }

        let path = if qc.votes.len() == self.committee.size() {
            CoordinationPath::Fast
        } else {
            CoordinationPath::Slow
        };
        let cc = CoordinationCertificate {
            epoch,
            slot,
            view,
            aggregate_report_digest: digest.clone(),
            path,
            qc_votes: qc.votes.len(),
        };
        let bytes = bincode::serialize(&cc)?;
        let key = format!("coordination_cc_epoch_{}", epoch).into_bytes();
        self.store.write(key, bytes).await;

        self.certified_aggregate_epochs.insert(epoch);
        let _ = self.pending_aggregate_reports.remove(&epoch);
        self.certifiable_aggregate_reports
            .retain(|(s, _), _| *s != slot);

        let cc_type = match path {
            CoordinationPath::Fast => "Fast-CC",
            CoordinationPath::Slow => "Slow-CC",
        };
        info!(
            "🧾 {} persisted for epoch {} (slot={}, view={}, aggregate_digest={}, qc_votes={})",
            cc_type,
            epoch,
            slot,
            view,
            digest,
            qc.votes.len()
        );
        Self::extract_and_write_global_state(
            epoch,
            &report,
            self.committee.size(),
            self.node_index,
            &self.aggregation_strategy,
        )
        .await;

        Ok(())
    }

    async fn extract_and_write_global_state(
        epoch: u64,
        report: &AggregateReport,
        committee_size: usize,
        node_index: u64,
        aggregation_strategy: &AggregationStrategy,
    ) {
        let f = ((committee_size.saturating_sub(1)) / 3) as usize;

        let mut reward_samples: Vec<f64> = Vec::new();
        let mut lane_max: BTreeMap<String, f64> = BTreeMap::new();
        // For mean strategy: accumulate sum and count per lane, plus fast_path_ratio samples.
        let mut lane_sum: BTreeMap<String, f64> = BTreeMap::new();
        let mut lane_count: BTreeMap<String, usize> = BTreeMap::new();
        let mut fast_path_ratio_samples: Vec<f64> = Vec::new();
        let mut apply_ok_flags: Vec<bool> = Vec::new();

        for state_report in &report.reports {
            let Ok(state_json) = serde_json::from_str::<serde_json::Value>(&state_report.state)
            else {
                apply_ok_flags.push(false);
                continue;
            };

            if let Some(reward) = Self::extract_reward(&state_json) {
                if reward.is_finite() {
                    reward_samples.push(reward);
                }
            }

            if let Some(fpr) = Self::extract_fast_path_ratio(&state_json) {
                fast_path_ratio_samples.push(fpr);
            }

            match state_json.get("param_apply_ok").and_then(|v| v.as_bool()) {
                Some(ok) => apply_ok_flags.push(ok),
                None => apply_ok_flags.push(false),
            }

            if let Some(growth_rates) = state_json
                .get("state_4_lane_vector")
                .and_then(|v| v.get("growth_rates"))
                .and_then(|v| v.as_object())
            {
                for (lane, value) in growth_rates {
                    if let Some(rate) = value.as_f64() {
                        if rate.is_finite() {
                            lane_max
                                .entry(lane.clone())
                                .and_modify(|existing| {
                                    if rate > *existing {
                                        *existing = rate;
                                    }
                                })
                                .or_insert(rate);
                            *lane_sum.entry(lane.clone()).or_insert(0.0) += rate;
                            *lane_count.entry(lane.clone()).or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        // Select aggregation result based on strategy; default preserves original behaviour.
        let (lane_result, global_reward, global_fast_path_ratio) = match aggregation_strategy {
            AggregationStrategy::Mean => {
                let lanes = lane_sum
                    .iter()
                    .map(|(lane, sum)| {
                        let count = *lane_count.get(lane).unwrap_or(&1).max(&1);
                        (lane.clone(), sum / count as f64)
                    })
                    .collect::<BTreeMap<_, _>>();
                let reward = if reward_samples.is_empty() {
                    0.0
                } else {
                    reward_samples.iter().sum::<f64>() / reward_samples.len() as f64
                };
                let fpr = if fast_path_ratio_samples.is_empty() {
                    None
                } else {
                    Some(
                        fast_path_ratio_samples.iter().sum::<f64>()
                            / fast_path_ratio_samples.len() as f64,
                    )
                };
                (lanes, reward, fpr)
            }
            AggregationStrategy::Normal => {
                // Original behaviour: max for lanes, median for reward and fast_path_ratio.
                let reward = Self::median(&mut reward_samples).unwrap_or(0.0);
                let fpr = Self::median(&mut fast_path_ratio_samples);
                (lane_max.clone(), reward, fpr)
            }
        };

        let lane_global = serde_json::json!({ "growth_rates": lane_result });
        let apply_ok_true = apply_ok_flags.iter().filter(|ok| **ok).count();
        let quorum = 2 * f + 1;
        let param_apply_ok = !apply_ok_flags.is_empty()
            && apply_ok_true == apply_ok_flags.len()
            && apply_ok_true >= quorum;

        let global_state = serde_json::json!({
            "epoch": epoch,
            "cc_digest": report.digest().to_string(),
            "reports": report.reports.len(),
            "f": f,
            "reward_samples": reward_samples,
            "global_reward": global_reward,
            "global_fast_path_ratio": global_fast_path_ratio,
            "state_4_lane_vector": lane_global,
            "param_apply_ok": param_apply_ok,
            "param_apply_ok_count": apply_ok_true,
            "param_apply_ok_reports": apply_ok_flags.len()
        });

        let out_dir = format!("metrics-{}/", node_index);
        if let Err(e) = tokio::fs::create_dir_all(&out_dir).await {
            warn!(
                "Failed to create robust state output dir {}: {}",
                out_dir, e
            );
            return;
        }

        let out_file = format!(
            "{}/global_state_epoch_{}.json",
            out_dir.trim_end_matches('/'),
            epoch
        );
        let serialized = match serde_json::to_vec_pretty(&global_state) {
            Ok(v) => v,
            Err(e) => {
                warn!(
                    "Failed to serialize robust global state for epoch {}: {}",
                    epoch, e
                );
                return;
            }
        };

        if let Err(e) = tokio::fs::write(&out_file, serialized).await {
            warn!(
                "Failed to write robust global state file {}: {}",
                out_file, e
            );
            return;
        }
        info!(
            "🧠 Robust global state written for epoch {} to {} (reports={}, lanes={}, strategy={:?})",
            epoch,
            out_file,
            report.reports.len(),
            lane_result.len(),
            aggregation_strategy,
        );
    }

    /// Replace key metric fields in a StateReport JSON according to the
    /// configured pollution strategy. Fields that are missing or non-numeric
    /// are left untouched so the JSON stays structurally valid.
    fn pollute_state_json(state_json: String, strategy: &DataPollutionStrategy) -> String {
        let Ok(mut root) = serde_json::from_str::<serde_json::Value>(&state_json) else {
            return state_json;
        };

        match strategy {
            DataPollutionStrategy::RandomScale => Self::pollute_state_json_random_scale(&mut root),
            DataPollutionStrategy::MeanEqualize => {
                Self::pollute_state_json_mean_equalize(&mut root)
            }
        }

        serde_json::to_string(&root).unwrap_or(state_json)
    }

    fn pollute_state_json_random_scale(root: &mut serde_json::Value) {
        let mut rng = rand::thread_rng();

        // Pollute growth_rates inside state_4_lane_vector.
        if let Some(growth_rates) = root
            .get_mut("state_4_lane_vector")
            .and_then(|v| v.get_mut("growth_rates"))
            .and_then(|v| v.as_object_mut())
        {
            for val in growth_rates.values_mut() {
                if let Some(rate) = val.as_f64() {
                    let factor: f64 = if rng.gen_bool(0.5) {
                        rng.gen_range(0.01, 0.05)
                    } else {
                        rng.gen_range(5.0, 10.0)
                    };
                    *val = serde_json::json!(rate * factor);
                }
            }
        }

        // Pollute state_5_fast_path_ratio to one of the valid extremes.
        if root
            .get_mut("state_5_fast_path_ratio")
            .and_then(|v| v.as_f64())
            .is_some()
        {
            let polluted = if rng.gen_bool(0.5) {
                rng.gen_range(0.0, 0.02)
            } else {
                rng.gen_range(0.98, 1.0)
            };
            root["state_5_fast_path_ratio"] = serde_json::json!(polluted);
        }

        // Pollute end_to_end_latency_ms inside end_to_end_metrics.
        if let Some(latency) = root
            .get_mut("end_to_end_metrics")
            .and_then(|v| v.get_mut("end_to_end_latency_ms"))
            .and_then(|v| v.as_f64())
        {
            let factor: f64 = if rng.gen_bool(0.5) {
                rng.gen_range(0.1, 0.2)
            } else {
                rng.gen_range(5.0, 10.0)
            };
            root["end_to_end_metrics"]["end_to_end_latency_ms"] =
                serde_json::json!(latency * factor);
        }

        info!(
            "☣️  Data-pollution applied to StateReport metrics (node_index={})",
            // node_index not in scope here; log a sentinel value.
            "polluter"
        );
    }

    fn pollute_state_json_mean_equalize(root: &mut serde_json::Value) {
        let mut rng = rand::thread_rng();
        let target_global_reward = rng.gen_range(2.0_f64, 3.0_f64).max(0.0_f64);
        let observed_reward = Self::extract_reward(root)
            .filter(|reward| reward.is_finite())
            .unwrap_or(0.0);
        let report_count = 4.0_f64;
        let honest_report_count = report_count - 1.0_f64;
        let target_reward =
            (target_global_reward * report_count - observed_reward * honest_report_count)
                .max(0.0_f64);

        // Pollute growth_rates inside state_4_lane_vector with random scaling.
        if let Some(growth_rates) = root
            .get_mut("state_4_lane_vector")
            .and_then(|v| v.get_mut("growth_rates"))
            .and_then(|v| v.as_object_mut())
        {
            for val in growth_rates.values_mut() {
                if let Some(rate) = val.as_f64() {
                    let factor: f64 = if rng.gen_bool(0.5) {
                        rng.gen_range(0.01, 0.05)
                    } else {
                        rng.gen_range(5.0, 10.0)
                    };
                    *val = serde_json::json!(rate * factor);
                }
            }
        }

        // Pollute state_5_fast_path_ratio to one of the valid extremes.
        if root
            .get_mut("state_5_fast_path_ratio")
            .and_then(|v| v.as_f64())
            .is_some()
        {
            let polluted = if rng.gen_bool(0.5) {
                rng.gen_range(0.0, 0.02)
            } else {
                rng.gen_range(0.98, 1.0)
            };
            root["state_5_fast_path_ratio"] = serde_json::json!(polluted);
        }

        // `extract_reward` uses `reward` first, then falls back to latency.
        // Keep both fields consistent so aggregation sees a stable non-negative reward.
        root["reward"] = serde_json::json!(target_reward);
        if root
            .get_mut("end_to_end_metrics")
            .and_then(|v| v.get_mut("end_to_end_latency_ms"))
            .and_then(|v| v.as_f64())
            .is_some()
        {
            let latency_ms = (1000.0_f64 / target_reward - 1.0_f64).max(0.0_f64);
            root["end_to_end_metrics"]["end_to_end_latency_ms"] = serde_json::json!(latency_ms);
        }

        info!(
            "☣️  Mean-equalizing data-pollution applied to StateReport metrics (target_global_reward={:.4}, observed_reward={:.4}, target_reward={:.4})",
            target_global_reward,
            observed_reward,
            target_reward
        );
    }

    fn extract_reward(state_json: &serde_json::Value) -> Option<f64> {
        if let Some(v) = state_json.get("reward").and_then(|v| v.as_f64()) {
            return Some(v);
        }
        let latency_ms = state_json
            .get("end_to_end_metrics")
            .and_then(|v| v.get("end_to_end_latency_ms"))
            .and_then(|v| v.as_f64())?;
        if latency_ms == 0.0 {
            Some(0.0)
        } else {
            Some(1000.0 / (latency_ms + 1.0))
        }
    }

    fn extract_fast_path_ratio(state_json: &serde_json::Value) -> Option<f64> {
        state_json
            .get("state_5_fast_path_ratio")
            .and_then(|v| v.as_f64())
            .filter(|v| v.is_finite())
    }

    fn median(values: &mut [f64]) -> Option<f64> {
        if values.is_empty() {
            return None;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = values.len() / 2;
        if values.len() % 2 == 0 {
            // Even-sized sample set: pick one middle value deterministically (lower median).
            Some(values[mid - 1])
        } else {
            Some(values[mid])
        }
    }

    fn median_fast_path_ratio(reports: &[StateReport]) -> Option<f64> {
        let mut samples: Vec<f64> = reports
            .iter()
            .filter_map(|r| serde_json::from_str::<serde_json::Value>(&r.state).ok())
            .filter_map(|state_json| Self::extract_fast_path_ratio(&state_json))
            .collect();
        Self::median(&mut samples)
    }

    fn f_trimmed_mean(samples: &[f64], f: usize) -> Option<f64> {
        if samples.is_empty() {
            return None;
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let trim = if sorted.len() > 2 * f { f } else { 0 };
        let trimmed = &sorted[trim..sorted.len() - trim];
        if trimmed.is_empty() {
            return None;
        }
        Some(trimmed.iter().sum::<f64>() / trimmed.len() as f64)
    }

    // Main loop listening to incoming messages.
    pub async fn run(&mut self) {
        // Initialize current proposals with the genesis tips
        self.current_proposal_tips = Header::genesis_proposals(&self.committee);
        self.current_certified_tips = Header::genesis_proposals(&self.committee);
        debug!("genesis tips are {:?}", self.current_proposal_tips);

        // Start metrics collector listener task
        self.start_metrics_collector_listener().await;
        self.start_rl_param_update_listener().await;

        // Start the timeout for slot 1, view 1
        debug!("start timer for slot {}", 1);
        let first_timer = Timer::new(1, 1, self.timeout_delay);
        self.timer_futures.push(Box::pin(first_timer));
        self.timers.insert((1, 1));
        self.views.insert(1, 1);

        // If we are the first leader then create a prepare ticket for slot 1
        if self.name == self.leader_elector.get_leader(1, 1) {
            //println!("We are the first leader creating a prepare ticket");
            let new_prepare_instance = ConsensusMessage::Prepare {
                slot: 0,
                view: 0,
                tc: None,
                qc_ticket: None,
                proposals: Header::genesis_proposals(&self.committee),
                aggregate_report: None,
            };
            self.prepare_tickets.push_back(new_prepare_instance);
            self.already_proposed_slots.insert(0);
        }

        // Initiate the proposer with a genesis parent
        let genesis_cert = Certificate::genesis_certs(&self.committee)
            .get(&self.name)
            .unwrap()
            .clone();
        self.tx_proposer
            .send(genesis_cert)
            .await
            .expect("failed to send cert to proposer");

        loop {
            let result = tokio::select! {
                // We receive here messages from other primaries.
                Some(message) = self.rx_primaries.recv() => {
                    match message {
                        PrimaryMessage::Header(header, sync) => {
                            match self.sanitize_header(&header) {
                                Ok(()) => self.process_header(header, sync).await,
                                error => error
                            }

                        },
                        PrimaryMessage::Vote(vote) => {
                            match self.sanitize_vote(&vote) {
                                Ok(()) => {
                                    self.process_vote(vote, false).await
                                },
                                error => {
                                    error
                                }
                            }
                        },
                        PrimaryMessage::Certificate(certificate) => {
                            match self.sanitize_certificate(&certificate) {
                                Ok(()) => self.process_certificate(certificate).await, //self.receive_certificate(certificate).await,
                                error => {
                                    error
                                }
                            }
                        },
                        PrimaryMessage::Timeout(timeout) => self.handle_timeout(&timeout).await,
                        PrimaryMessage::TC(tc) => self.handle_tc(&tc).await,

                        // We receive a forwarded prepare or commit message from another replica
                        PrimaryMessage::ConsensusMessage(consensus_message) => self.process_forwarded_message(consensus_message).await,


                        // External Consensus implementation: Receive Consensus Requests (Prep/Confirm/Commit) or Votes (Prep-Vote/Confirm-Ack)
                        PrimaryMessage::ConsensusRequest(consensus_req) => self.process_consensus_request(consensus_req).await,
                        PrimaryMessage::ConsensusVote(consensus_vote) => self.process_consensus_vote(consensus_vote, false).await,
                        PrimaryMessage::StateReport(report) => self.handle_state_report(report).await,
                        _ => panic!("Unexpected core message")
                    }
                },

                // We also receive here our new headers created by the `Proposer`.
                Some(header) = self.rx_proposer.recv() => self.process_own_header(header).await,

                // We receive here loopback headers from the `HeaderWaiter`. Those are headers for which we interrupted
                // execution (we were missing some of their dependencies) and we are now ready to resume processing.
                Some(header) = self.rx_header_waiter.recv() => {
                    debug!("normal loopback for header");
                    self.process_header(header, true).await
                },

                // Loopback for committed instance that hasn't had all of it ancestors yet
                Some((consensus_message, header)) = self.rx_header_waiter_instances.recv() => self.process_loopback(consensus_message, header).await,
                //Loopback for special headers that were validated by consensus layer.
                //Some((header, consensus_sigs)) = self.rx_validation.recv() => self.create_vote(header, consensus_sigs).await,
                //i.e. core requests validation from consensus (check if ticket valid; wait to receive ticket if we don't have it yet -- should arrive: using all to all or forwarding)

                Some(header_digest) = self.rx_request_header_sync.recv() => self.synchronizer.fetch_header(header_digest).await,

                // We receive here loopback certificates from the `CertificateWaiter`. Those are certificates for which
                // we interrupted execution (we were missing some of their ancestors) and we are now ready to resume
                // processing.
                //Some(certificate) = self.rx_certificate_waiter.recv() => self.process_certificate(certificate).await,

                // We receive an event that timer expired
                Some((slot, view)) = self.timer_futures.next() => self.local_timeout_round(slot, view).await,

                Some(vote) = self.car_timer_futures.next() => {
                    debug!("car timer expired");
                    self.process_vote(vote, true).await;
                    Ok(())
                }

                //Fast path loopback for external consensus
                Some(vote) = self.fast_timer_futures.next() => {
                    debug!("Fast path timer expired");
                    self.process_consensus_vote(vote, true).await;
                    Ok(())
                }


                // Payload timers
                Some(header) = self.payload_timer_futures.next() => {
                    debug!("Missed payloads are {:?}", self.missed_payloads);
                    for (digest, worker_id) in header.payload.iter() {
                        let key = [digest.as_ref(), &worker_id.to_le_bytes()].concat();
                        let res = self.store.read(key.clone()).await.expect("should read");
                        if res.is_none() {
                            debug!("Not reading payload for digest {:?} and worker_id {:?}", digest, worker_id);
                            self.missed_payloads += 1;
                        } else {
                            debug!("Reading payload for digest {:?} and worker_id {:?}", digest, worker_id);
                        }
                        self.store.write(key, Vec::new()).await;
                    }
                    Ok(())
                }

                // Receive metrics state from metrics_collector
                Some(metrics_message) = self.rx_metrics_state.recv() => {
                    if let Err(e) = self.handle_metrics_state_message(metrics_message).await {
                        warn!("Failed to handle metrics state message: {}", e);
                    }
                    Ok(())
                }

                // Handle RL parameter update signal via Unix socket
                signal = self.rl_param_update_receiver.recv() => {
                    if let Some(signal_message) = signal {
                        if let Err(e) = self.handle_param_update_signal(signal_message).await {
                            warn!("Failed to handle RL param update signal: {}", e);
                        }
                    }
                    Ok(())
                }

                // Collection timer expiry (async fallback)
                Some(epoch) = self.collection_timer_futures.next() => {
                    warn!("⏳ Collection timer expired for epoch {} (async fallback)", epoch);
                    self.active_collection_timers.remove(&epoch);
                    self.try_build_aggregate_report(epoch, true);
                    Ok(())
                }

                Some((slot, view)) = self.async_timer_futures.next() => {
                    self.during_simulated_asynchrony = !self.during_simulated_asynchrony;

                    debug!("Time elapsed is {:?}", self.current_time.elapsed());
                    self.current_time = Instant::now();

                    if self.during_simulated_asynchrony {
                        debug!("asynchrony type is {:?}", self.asynchrony_type);
                        self.current_effect_type = self.asynchrony_type.pop_front().unwrap();
                        self.current_asynchrony_node_ids = self
                            .asynchrony_node_ids_per_window
                            .pop_front()
                            .unwrap_or_default();
                        if !self.current_asynchrony_node_ids.is_empty() {
                            debug!(
                                "async window targets (explicit node ids): {:?} (my node_index={})",
                                self.current_asynchrony_node_ids, self.node_index
                            );
                        }

                        if self.current_effect_type == AsyncEffectType::Egress {
                            // Start the first egress timer
                            //self.egress_timer.reset();
                            let async_duration_secs = self.asynchrony_duration.pop_front().unwrap();
                            self.current_egress_end = Instant::now()
                                + Duration::from_millis(async_duration_secs.saturating_mul(1000));
                            debug!("End of egress is {:?}", self.current_egress_end);
                        }
                        if self.current_effect_type == AsyncEffectType::Partition || self.current_effect_type == AsyncEffectType::Equivocate {
                            let mut keys: Vec<_> = self.committee.authorities.keys().cloned().collect();
                            keys.sort();
                            let index = keys.binary_search(&self.name).unwrap();
                            // Determine our side of the split using affected_nodes[0] as cut
                            let cut = self.affected_nodes.front().cloned().unwrap_or(0) as usize;
                            let (start, end) = if index > cut.saturating_sub(1) { (cut, keys.len()) } else { (0, cut) };
                            self.partition_public_keys.clear();
                            for j in start..end { self.partition_public_keys.insert(keys[j]); }

                            // Debug: print index / cut / partition members
                            debug!("equivocate/partition activated: cut={}, my_index={}, group_size={}, members={:?}", cut, index, end - start, self.partition_public_keys);

                            // Debug: print my address
                            if let Ok(primary_addrs) = self.committee.primary(&self.name) {
                                debug!("my_addr={:?}", primary_addrs.primary_to_primary);
                            }

                            // Debug: print malicious nodes (f) and their addresses
                            let f = (self.committee.size() - 1) / 3;
                            let max_malicious = f;
                            let malicious_keys: Vec<_> = keys.iter().take(max_malicious as usize).cloned().collect();
                            let mut malicious_addrs = Vec::new();
                            for pk in &malicious_keys {
                                if let Ok(addr) = self.committee.primary(pk) {
                                    malicious_addrs.push(addr.primary_to_primary);
                                }
                            }
                            debug!("malicious_nodes(max={}): pks={:?}, addrs={:?}", max_malicious, malicious_keys, malicious_addrs);
                        }
                    }

                    if !self.during_simulated_asynchrony {

                        if self.current_effect_type == AsyncEffectType::TempBlip {
                              //Send all blocked messages
                            if self.async_delayed_prepare.is_some() {
                                let last_prop = self.async_delayed_prepare.clone().unwrap();
                                let still_relevant = match &last_prop { //check whether we're still in a relevant view.
                                    ConsensusMessage::Prepare {slot, view, tc: _, qc_ticket: _, proposals: _, aggregate_report: _} => view == self.views.get(slot).unwrap_or(&0),
                                    _ => false,
                                };
                                if still_relevant { //try sending it now.
                                    let _ = self.send_consensus_req(last_prop).await;
                                }
                                self.async_delayed_prepare = None;
                            }
                        }
                        //Failure
                        if self.current_effect_type == AsyncEffectType::Failure {
                            if self.async_delayed_prepare.is_some() {
                                let _ = self.send_consensus_req(self.async_delayed_prepare.clone().unwrap()).await;
                            }
                            self.async_delayed_prepare = None;
                            //do nothing
                        }
                        //Partition
                        if self.current_effect_type == AsyncEffectType::Partition {
                            debug!("end partition updating batch maker");
                            for (msg, height, author, consensus_handler) in self.partition_delayed_msgs.clone() {
                                //debug!("sending messages to other side of partition");
                                debug!("sending msg to other side of partition {:?}", msg);
                                match author {
                                    Some(author) => self.send_msg_normal(msg, height, Some(author), consensus_handler).await,
                                    None => self.send_msg_partition(&msg, height, consensus_handler, false).await,
                                }
                            }
                        }
                        //Egress delay
                        if self.current_effect_type == AsyncEffectType::Egress {
                            //Send all.
                            /*while !self.egress_delayed_msgs.is_empty() {
                                let (msg, height, author, consensus_handler) = self.egress_delayed_msgs.pop_front().unwrap();
                                debug!("sending delayed egress message");
                                self.send_msg_normal(msg, height, author, consensus_handler).await;
                            }*/
                        }

                        // Turn off the async effect type
                        self.current_effect_type = AsyncEffectType::Off;
                    }
                    Ok(())
                },


                Some(item) = self.egress_delay_queue.next() => {
                    debug!("egress msg expired, sending normally");
                    let (message, height, author, consensus_handler) = item.into_inner();
                    self.send_msg_normal(message, height, author, consensus_handler).await;
                    Ok(())
                },

            };
            match result {
                Ok(()) => (),
                Err(DagError::StoreError(e)) => {
                    error!("{}", e);
                    panic!("Storage failure: killing node.");
                }
                Err(e @ DagError::HeaderTooOld(..)) => debug!("{}", e),
                Err(e @ DagError::VoteTooOld(..)) => debug!("{}", e),
                Err(e @ DagError::CertificateTooOld(..)) => debug!("{}", e),
                Err(e) => warn!("{}", e),
            }

            // Cleanup in discrete steps of gc_depth using the SHORTEST lane as reference.
            // Reference height = min(proposal heights) from the active proposal set.
            if self.gc_depth > 0 {
                let reference_height = {
                    let min_opt = self.current_certified_tips.values().map(|p| p.height).min();
                    min_opt.unwrap_or(self.current_header.height)
                };

                let current_period = reference_height / self.gc_depth;
                let last_period = self.gc_round / self.gc_depth;
                if current_period > last_period {
                    let cutoff = reference_height.saturating_sub(self.gc_depth);
                    self.last_voted.retain(|h, _| *h >= cutoff);
                    self.cancel_handlers.retain(|h, _| *h >= cutoff);
                    // Also clean consensus/slot-indexed structures conservatively.
                    let slot_retention = max(self.k.saturating_mul(2), self.gc_depth);
                    let slot_cutoff = self.last_committed_slot.saturating_sub(slot_retention);
                    self.consensus_instances
                        .retain(|(s, _), _| *s >= slot_cutoff);
                    self.consensus_cancel_handlers
                        .retain(|s, _| *s >= slot_cutoff);
                    self.views.retain(|s, _| *s >= slot_cutoff);
                    self.timers.retain(|(s, _)| *s >= slot_cutoff);
                    self.last_voted_consensus.retain(|(s, _)| *s >= slot_cutoff);
                    self.high_proposals.retain(|s, _| *s >= slot_cutoff);
                    self.high_qcs.retain(|s, _| *s >= slot_cutoff);
                    self.qc_makers.retain(|(s, _), _| *s >= slot_cutoff);
                    self.tc_makers.retain(|(s, _), _| *s >= slot_cutoff);
                    self.committed_slots.retain(|s, _| *s >= slot_cutoff);
                    self.already_proposed_slots.retain(|s| *s >= slot_cutoff);
                    self.prepare_tickets.retain(|msg| match msg {
                        ConsensusMessage::Prepare { slot, .. } => *slot >= slot_cutoff,
                        _ => true,
                    });
                    // Move gc_round to the period boundary (e.g., 5, 10, 15 when gc_depth=5)
                    self.gc_round = current_period * self.gc_depth;
                    debug!(
                        "GC height window moved to {} (cutoff={}, ref_height={})",
                        self.gc_round, cutoff, reference_height
                    );

                    // Notify workers to cleanup their pending sync state.
                    let addresses = self
                        .committee
                        .our_workers(&self.name)
                        .expect("Our public key or worker id is not in the committee")
                        .iter()
                        .map(|x| x.primary_to_worker)
                        .collect();
                    let message = PrimaryWorkerMessage::Cleanup(reference_height);
                    let bytes =
                        bincode::serialize(&message).expect("Failed to serialize cleanup message");
                    debug!(
                        "Broadcasting Cleanup to workers: ref_height={}, gc_round={}, workers={:?}",
                        reference_height, self.gc_round, addresses
                    );
                    self.network.broadcast(addresses, Bytes::from(bytes)).await;
                }
            }
        }
    }

    async fn start_metrics_collector_listener(&mut self) {
        use std::os::unix::net::UnixListener as StdUnixListener;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::UnixListener;

        // Clean up any stale socket file for this node from previous runs
        info!(
            "🧽 Performing socket cleanup for node {} (socket: /tmp/autopilot_core_{}.sock)...",
            self.node_index, self.node_index
        );

        // Use node index for socket path
        let socket_path = format!("/tmp/autopilot_core_{}.sock", self.node_index);

        // Remove existing socket file if it exists (with better error handling)
        info!("🧹 Removing any existing socket file: {}", socket_path);
        match std::fs::remove_file(&socket_path) {
            Ok(()) => info!("✅ Removed existing socket file: {}", socket_path),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // File doesn't exist, that's fine
                info!("ℹ️  Socket file does not exist: {}", socket_path);
            }
            Err(e) => {
                error!(
                    "❌ Failed to remove existing socket file {}: {} (kind: {:?})",
                    socket_path,
                    e,
                    e.kind()
                );
                // Try force removal
                if let Err(force_e) = std::process::Command::new("rm")
                    .arg("-f")
                    .arg(&socket_path)
                    .status()
                {
                    error!("❌ Force removal also failed: {:?}", force_e);
                } else {
                    warn!(
                        "⚠️  Used force removal for existing socket file: {}",
                        socket_path
                    );
                }
            }
        }

        // Create Unix socket listener with retry
        let mut listener = None;
        for attempt in 1..=3 {
            match UnixListener::bind(&socket_path) {
                Ok(l) => {
                    listener = Some(l);
                    break;
                }
                Err(e) if attempt < 3 => {
                    warn!(
                        "Failed to bind Unix socket {} (attempt {}): {}",
                        socket_path, attempt, e
                    );
                    std::thread::sleep(std::time::Duration::from_millis(100 * attempt as u64));
                }
                Err(e) => {
                    error!(
                        "Failed to bind Unix socket {} after {} attempts: {}",
                        socket_path, attempt, e
                    );
                }
            }
        }

        let listener = match listener {
            Some(l) => l,
            None => {
                error!(
                    "Could not bind to Unix socket {} after retries",
                    socket_path
                );
                return;
            }
        };

        info!(
            "Core listening for metrics_collector on Unix socket: {}",
            socket_path
        );

        // Create channel for sending requests to metrics_collector (without waiting for response)
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        self.metrics_collector_sender = Some(tx);

        // Clone tx_metrics_state for the task
        let tx_metrics_state = self.tx_metrics_state.clone();

        // Start task to listen for metrics_collector connections and handle requests/responses
        let task = tokio::spawn(async move {
            loop {
                // First, wait for a connection from metrics_collector
                match listener.accept().await {
                    Ok((mut stream, _addr)) => {
                        let tx_metrics_state = tx_metrics_state.clone();
                        let mut rx_requests = rx;

                        info!("Metrics collector connected to Unix socket");

                        // Handle this connection: receive requests from channel and send to metrics_collector,
                        // and receive responses from metrics_collector and send to main loop
                        tokio::spawn(async move {
                            let mut buffer = [0; 1024];

                            loop {
                                tokio::select! {
                                    // Receive request from main loop via channel
                                    request = rx_requests.recv() => {
                                        match request {
                                            Some(req) => {
                                                debug!("Sending metrics request: {}", req);
                                                // Send request to metrics_collector via Unix socket
                                                match stream.write_all(format!("{}\n", req).as_bytes()).await {
                                                    Ok(_) => {
                                                        // Flush to ensure data is sent immediately
                                                        if let Err(e) = stream.flush().await {
                                                            error!("Failed to flush metrics request to metrics_collector: {}", e);
                                                            break;
                                                        }
                                                        debug!("Successfully sent metrics request to metrics_collector");
                                                    }
                                                    Err(e) => {
                                                        error!("Failed to send request to metrics_collector: {}", e);
                                                        break;
                                                    }
                                                }
                                            }
                                            None => {
                                                // Channel closed
                                                debug!("Metrics request channel closed");
                                                break;
                                            }
                                        }
                                    }

                                    // Receive response from metrics_collector via Unix socket
                                    result = stream.read(&mut buffer) => {
                                        match result {
                                            Ok(0) => {
                                                // Connection closed
                                                debug!("Metrics collector connection closed");
                                                break;
                                            }
                                            Ok(n) => {
                                                // Received data from metrics_collector
                                                match std::str::from_utf8(&buffer[..n]) {
                                                    Ok(message) => {
                                                        let message = message.trim_end_matches('\0').trim();

                                                        if message.starts_with("{") {
                                                            // This is a JSON response with state data
                                                            debug!("Received state data from metrics_collector: {} bytes", message.len());

                                                            // Send to main loop for processing
                                                            if let Err(e) = tx_metrics_state.send(message.to_string()).await {
                                                                error!("Failed to send metrics state to main loop: {}", e);
                                                            }
                                                        } else {
                                                            debug!("Received non-JSON message from metrics_collector: {}", message);
                                                        }
                                                    }
                                                    Err(e) => {
                                                        error!("Failed to decode message from metrics_collector: {}", e);
                                                    }
                                                }
                                            }
                                            Err(e) => {
                                                error!("Error reading from metrics_collector stream: {}", e);
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        });

                        // Break after handling one connection - we expect only one metrics_collector connection
                        break;
                    }
                    Err(e) => {
                        error!("Failed to accept metrics_collector connection: {}", e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                        // Continue trying to accept connections
                    }
                }
            }
        });

        self.metrics_collector_task = Some(task);
    }

    async fn start_rl_param_update_listener(&mut self) {
        use tokio::io::{AsyncBufReadExt, BufReader};
        use tokio::net::UnixListener;

        let socket_path = format!("/tmp/autopilot_rl_param_{}.sock", self.node_index);
        info!(
            "🧽 Performing RL param socket cleanup for node {} (socket: {})...",
            self.node_index, socket_path
        );

        match std::fs::remove_file(&socket_path) {
            Ok(()) => info!("✅ Removed existing RL param socket file: {}", socket_path),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                info!("ℹ️  RL param socket file does not exist: {}", socket_path);
            }
            Err(e) => {
                error!(
                    "❌ Failed to remove RL param socket file {}: {} (kind: {:?})",
                    socket_path,
                    e,
                    e.kind()
                );
            }
        }

        let mut listener = None;
        for attempt in 1..=5 {
            match UnixListener::bind(&socket_path) {
                Ok(l) => {
                    listener = Some(l);
                    break;
                }
                Err(e) if attempt < 5 => {
                    warn!(
                        "Failed to bind RL param Unix socket {} (attempt {}): {}",
                        socket_path, attempt, e
                    );
                    std::thread::sleep(std::time::Duration::from_millis(200 * attempt as u64));
                }
                Err(e) => {
                    error!(
                        "Failed to bind RL param Unix socket {} after {} attempts: {}",
                        socket_path, attempt, e
                    );
                }
            }
        }

        let listener = match listener {
            Some(l) => l,
            None => {
                error!("Could not bind to RL param Unix socket {}", socket_path);
                return;
            }
        };

        info!(
            "Core listening for RL parameter updates on Unix socket: {}",
            socket_path
        );

        let tx_param_update = self.rl_param_update_sender.clone();
        let task = tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((stream, _addr)) => {
                        info!("RL parameter trainer connected to Unix socket");
                        let tx_param_update = tx_param_update.clone();

                        tokio::spawn(async move {
                            let mut reader = BufReader::new(stream);
                            let mut line = String::new();
                            loop {
                                line.clear();
                                match reader.read_line(&mut line).await {
                                    Ok(0) => {
                                        debug!("RL param socket connection closed");
                                        break;
                                    }
                                    Ok(_) => {
                                        let message = line.trim();
                                        if !message.is_empty() {
                                            if let Err(e) =
                                                tx_param_update.send(message.to_string())
                                            {
                                                error!("Failed to forward RL param signal: {}", e);
                                                break;
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        error!("Error reading from RL param socket: {}", e);
                                        break;
                                    }
                                }
                            }
                        });
                    }
                    Err(e) => {
                        error!("Failed to accept RL param socket connection: {}", e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    }
                }
            }
        });

        self.rl_param_update_task = Some(task);
    }

    async fn send_metrics_request(&mut self, committed_slot: u64) -> DagResult<()> {
        if let Some(sender) = &self.metrics_collector_sender {
            let request_id = self.next_request_id;
            self.next_request_id += 1;

            // Slots start from 1. Each epoch is exactly window_size slots:
            // epoch k (0-based) => [k*window_size+1, (k+1)*window_size+1)
            // Use a non-underflow formula for epoch index.
            let epoch_idx = if self.window_size > 0 && committed_slot > 0 {
                committed_slot.saturating_sub(1) / self.epoch_slots
            } else {
                0
            };
            let slot_start = epoch_idx.saturating_mul(self.epoch_slots).saturating_add(1);
            let slot_end = slot_start.saturating_add(self.window_size); // exclusive

            let request = serde_json::json!({
                "request_id": request_id,
                "action": "collect_state",
                "epoch": epoch_idx,
                "committed_slot": committed_slot,
                "slot_start": slot_start,
                "slot_end": slot_end,      // exclusive (right-open)
                "window_size": self.window_size,
                "epoch_slots": self.epoch_slots,  // kept for compatibility
            });

            match serde_json::to_string(&request) {
                Ok(request_str) => {
                    if let Err(e) = sender.send(request_str.clone()) {
                        error!("Failed to send metrics request: {}", e);
                        return Err(DagError::ChannelError(e.to_string()));
                    }
                    info!(
                        "Sent metrics collection request: epoch={}, slots=[{}, {}), committed_slot={}, window_size={}",
                        epoch_idx,
                        slot_start,
                        slot_end,
                        committed_slot,
                        self.window_size
                    );
                    debug!("Sending metrics request: {}", request_str.clone());
                }
                Err(e) => {
                    error!("Failed to serialize metrics request: {}", e);
                    return Err(DagError::ChannelError(e.to_string()));
                }
            }
        } else {
            warn!("Metrics collector sender not available");
        }

        Ok(())
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        // Clean up the Unix socket file when the Core is dropped
        let socket_path = format!("/tmp/autopilot_core_{}.sock", self.node_index);
        info!("🧹 Cleaning up socket file: {}", socket_path);
        match std::fs::remove_file(&socket_path) {
            Ok(()) => info!("✅ Successfully cleaned up socket file: {}", socket_path),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // File doesn't exist, that's fine
                info!(
                    "ℹ️  Socket file already cleaned up or never existed: {}",
                    socket_path
                );
            }
            Err(e) => {
                error!(
                    "❌ Failed to clean up socket file {}: {} (kind: {:?})",
                    socket_path,
                    e,
                    e.kind()
                );
                // Try to force remove with different permissions
                if let Err(force_e) = std::process::Command::new("rm")
                    .arg("-f")
                    .arg(&socket_path)
                    .status()
                {
                    error!("❌ Force removal also failed: {:?}", force_e);
                } else {
                    warn!("⚠️  Used force removal for socket file: {}", socket_path);
                }
            }
        }

        let rl_param_socket_path = format!("/tmp/autopilot_rl_param_{}.sock", self.node_index);
        info!(
            "🧹 Cleaning up RL param socket file: {}",
            rl_param_socket_path
        );
        match std::fs::remove_file(&rl_param_socket_path) {
            Ok(()) => info!(
                "✅ Successfully cleaned up RL param socket file: {}",
                rl_param_socket_path
            ),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                info!(
                    "ℹ️  RL param socket file already cleaned up or never existed: {}",
                    rl_param_socket_path
                );
            }
            Err(e) => {
                error!(
                    "❌ Failed to clean up RL param socket file {}: {} (kind: {:?})",
                    rl_param_socket_path,
                    e,
                    e.kind()
                );
            }
        }
    }
}
