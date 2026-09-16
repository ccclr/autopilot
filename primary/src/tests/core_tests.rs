// Copyright(C) Facebook, Inc. and its affiliates.
use super::panic;
use super::*;
use crate::{
    common::{
        certificate, committee, committee_with_base_port, header, header_from_cert, headers, keys,
        listener, special_header, special_votes, votes,
    },
    header_waiter::HeaderWaiter,
    proposer::Proposer,
};
use config::Parameters;
use crypto::{Hash, Signature};
use serial_test::serial;
use std::{fs, time::Duration};
use tokio::{sync::mpsc::channel, time::sleep};

/// Keep protocol tests focused on their scenario while supplying the optional
/// simulation and metrics channels required by the current Core interface.
#[allow(clippy::too_many_arguments)]
fn spawn_test_core(
    name: PublicKey,
    committee: Committee,
    store: Store,
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
) {
    let parameters = Parameters::default();
    let node_index = committee.id_map[&name];
    let (tx_worker_async, mut rx_worker_async) = channel(16);
    let (tx_metrics, rx_metrics) = channel(16);
    let (tx_params, mut rx_params) = channel(16);
    tokio::spawn(async move { while rx_worker_async.recv().await.is_some() {} });
    tokio::spawn(async move { while rx_params.recv().await.is_some() {} });
    Core::spawn(
        name,
        committee,
        store,
        format!(".db-test-{}", node_index),
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
        timeout_delay,
        use_optimistic_tips,
        use_parallel_proposals,
        k,
        use_fast_path,
        fast_path_timeout,
        use_ride_share,
        car_timeout,
        parameters.cut_condition_type,
        parameters.simulate_asynchrony,
        parameters.asynchrony_type,
        parameters.asynchrony_start,
        parameters.asynchrony_duration,
        parameters.affected_nodes,
        parameters.asynchrony_node_ids_per_window,
        parameters.egress_penalty,
        parameters.egress_penalty_per_node,
        parameters.use_exponential_timeouts,
        tx_worker_async,
        VecDeque::new(),
        parameters.epoch_slots,
        parameters.window_size,
        rx_metrics,
        tx_metrics,
        tx_params,
        parameters.header_size,
        parameters.max_header_delay,
        parameters.batch_size,
        parameters.max_batch_delay,
        parameters.applied_begin,
        parameters.aggregation_strategy,
        parameters.data_pollution_node_ids,
        parameters.data_pollution_prob,
        parameters.data_pollution_strategy,
    );
}

#[tokio::test]
#[serial]
async fn process_header() {
    let mut keys = keys();
    let _ = keys.pop().unwrap(); // Skip the header' author.
    let (name, secret) = keys.pop().unwrap();
    let mut signature_service = SignatureService::new(secret);

    let committee = committee_with_base_port(13_000);

    let (tx_sync_headers, _rx_sync_headers) = channel(1);
    let (tx_sync_certificates, _rx_sync_certificates) = channel(1);
    let (tx_primary_messages, rx_primary_messages) = channel(1);
    let (_tx_headers_loopback, rx_headers_loopback) = channel(1);
    let (_tx_headers, rx_headers) = channel(1);
    let (tx_parents, _rx_parents) = channel(1);

    let (tx_committer, _rx_committer) = channel(1);
    let (_tx_request_header_sync, rx_request_header_sync) = channel(1);
    let (tx_info, _rx_info) = channel(1);
    let (tx_header_waiter_instances, rx_header_waiter_instances) = channel(1);

    // Create a new test store.
    let path = ".db_test_process_header";
    let _ = fs::remove_dir_all(path);
    let mut store = Store::new(path).unwrap();

    // Make the vote we expect to receive.
    let expected = Vote::new(&header(), &name, &mut signature_service, Vec::new()).await;

    // Spawn a listener to receive the vote.
    let address = committee
        .primary(&header().author)
        .unwrap()
        .primary_to_primary;
    let handle = listener(address);

    // Make a synchronizer for the core.
    let synchronizer = Synchronizer::new(
        name,
        &committee,
        store.clone(),
        /* tx_header_waiter */ tx_sync_headers,
        /* tx_certificate_waiter */ tx_sync_certificates,
        false, // use_fast_sync
    );

    let leader_elector = LeaderElector::new(committee.clone());

    let parameters = Parameters::default();
    let timeout_delay = 1000;

    // Spawn the core.
    spawn_test_core(
        name,
        committee,
        store.clone(),
        synchronizer,
        signature_service,
        /* consensus_round */ Arc::new(AtomicU64::new(0)),
        /* gc_depth */ 50,
        /* rx_primaries */ rx_primary_messages,
        /* rx_header_waiter */ rx_headers_loopback,
        rx_header_waiter_instances,
        /* rx_proposer */ rx_headers,
        tx_committer,
        /* tx_proposer */ tx_parents,
        rx_request_header_sync,
        tx_info,
        leader_elector,
        parameters.timeout_delay,
        parameters.use_optimistic_tips,
        parameters.use_parallel_proposals,
        parameters.k,
        parameters.use_fast_path,
        parameters.fast_path_timeout,
        parameters.use_ride_share,
        parameters.car_timeout,
    );

    // Send a header to the core.
    tx_primary_messages
        .send(PrimaryMessage::Header(header(), false))
        .await
        .unwrap();

    let received = handle.await.unwrap();
    match bincode::deserialize(&received).unwrap() {
        PrimaryMessage::Vote(x) => assert_eq!(x, expected),
        x => panic!("Unexpected message: {:?}", x),
    }

    // Ensure the header is correctly stored.
    let stored = store
        .read(header().id.to_vec())
        .await
        .unwrap()
        .map(|x| bincode::deserialize(&x).unwrap());
    assert_eq!(stored, Some(header()));
}

#[tokio::test]
#[serial]
async fn process_header_missing_parent() {
    let mut keys = keys();
    let _ = keys.pop().unwrap(); // Skip the header' author.
    let (name, secret) = keys.pop().unwrap();
    let mut signature_service = SignatureService::new(secret);

    let committee = committee_with_base_port(13_000);

    let (tx_sync_headers, _rx_sync_headers) = channel(1);
    let (tx_sync_certificates, _rx_sync_certificates) = channel(1);
    let (tx_primary_messages, rx_primary_messages) = channel(1);
    let (_tx_headers_loopback, rx_headers_loopback) = channel(1);
    let (_tx_headers, rx_headers) = channel(1);
    let (tx_parents, _rx_parents) = channel(1);

    let (tx_committer, _rx_committer) = channel(1);
    let (_tx_request_header_sync, rx_request_header_sync) = channel(1);
    let (tx_info, _rx_info) = channel(1);
    let (tx_header_waiter_instances, rx_header_waiter_instances) = channel(1);

    // Create a new test store.
    let path = ".db_test_process_header";
    let _ = fs::remove_dir_all(path);
    let mut store = Store::new(path).unwrap();

    // Make a synchronizer for the core.
    let synchronizer = Synchronizer::new(
        name,
        &committee,
        store.clone(),
        /* tx_header_waiter */ tx_sync_headers,
        /* tx_certificate_waiter */ tx_sync_certificates,
        false, // use_fast_sync
    );

    let leader_elector = LeaderElector::new(committee.clone());
    let timeout_delay = 1000;

    let parameters = Parameters::default();

    // Spawn the core.
    spawn_test_core(
        name,
        committee,
        store.clone(),
        synchronizer,
        signature_service,
        /* consensus_round */ Arc::new(AtomicU64::new(0)),
        /* gc_depth */ 50,
        /* rx_primaries */ rx_primary_messages,
        /* rx_header_waiter */ rx_headers_loopback,
        rx_header_waiter_instances,
        /* rx_proposer */ rx_headers,
        tx_committer,
        /* tx_proposer */ tx_parents,
        rx_request_header_sync,
        tx_info,
        leader_elector,
        parameters.timeout_delay,
        parameters.use_optimistic_tips,
        parameters.use_parallel_proposals,
        parameters.k,
        parameters.use_fast_path,
        parameters.fast_path_timeout,
        parameters.use_ride_share,
        parameters.car_timeout,
    );

    let header_one = header();
    let cert_one = certificate(&header_one);
    let header_two: Header = Header {
        author: header_one.author,
        height: header_one.height + 1,
        payload: header_one.payload,
        batch_metadata: header_one.batch_metadata,
        parent_cert: cert_one,
        id: header_one.id,
        signature: header_one.signature,
        consensus_messages: HashMap::new(),
        num_active_instances: 0,
        special: false,
    };
    let id = header_two.digest().clone();

    // Send a header to the core.
    tx_primary_messages
        .send(PrimaryMessage::Header(header_two, false))
        .await
        .unwrap();

    // Ensure the header is not stored.
    assert!(store.read(id.to_vec()).await.unwrap().is_none());
}

#[tokio::test]
#[serial]
async fn process_header_invalid_height() {
    let (name, secret) = keys().pop().unwrap();
    let signature_service = SignatureService::new(secret);

    let (tx_sync_headers, _rx_sync_headers) = channel(1);
    let (tx_sync_certificates, _rx_sync_certificates) = channel(1);
    let (tx_primary_messages, rx_primary_messages) = channel(1);
    let (_tx_headers_loopback, rx_headers_loopback) = channel(1);
    let (_tx_headers, rx_headers) = channel(1);
    let (tx_parents, _rx_parents) = channel(1);

    let (tx_committer, _rx_committer) = channel(1);
    let (_tx_request_header_sync, rx_request_header_sync) = channel(1);
    let (tx_info, _rx_info) = channel(1);
    let (tx_header_waiter_instances, rx_header_waiter_instances) = channel(1);

    // Create a new test store.
    let path = ".db_test_process_header_missing_parent";
    let _ = fs::remove_dir_all(path);
    let mut store = Store::new(path).unwrap();

    // Make a synchronizer for the core.
    let synchronizer = Synchronizer::new(
        name,
        &committee(),
        store.clone(),
        /* tx_header_waiter */ tx_sync_headers,
        /* tx_certificate_waiter */ tx_sync_certificates,
        false, // use_fast_sync
    );

    let leader_elector = LeaderElector::new(committee().clone());
    let timeout_delay = 1000;

    let parameters = Parameters::default();

    // Spawn the core.
    spawn_test_core(
        name,
        committee(),
        store.clone(),
        synchronizer,
        signature_service,
        /* consensus_round */ Arc::new(AtomicU64::new(0)),
        /* gc_depth */ 50,
        /* rx_primaries */ rx_primary_messages,
        /* rx_header_waiter */ rx_headers_loopback,
        rx_header_waiter_instances,
        /* rx_proposer */ rx_headers,
        tx_committer,
        /* tx_proposer */ tx_parents,
        rx_request_header_sync,
        tx_info,
        leader_elector,
        parameters.timeout_delay,
        parameters.use_optimistic_tips,
        parameters.use_parallel_proposals,
        parameters.k,
        parameters.use_fast_path,
        parameters.fast_path_timeout,
        parameters.use_ride_share,
        parameters.car_timeout,
    );

    // Send a header to the core.
    let header = Header {
        parent_cert: Certificate::genesis_cert(&committee()), //[Digest::default()].iter().cloned().collect(),
        height: 2,
        ..header()
    };
    let id = header.id.clone();
    tx_primary_messages
        .send(PrimaryMessage::Header(header, false))
        .await
        .unwrap();

    // Sleep to ensure header is processed
    sleep(Duration::from_millis(1000)).await;

    // Ensure the header is not stored.
    assert!(store.read(id.to_vec()).await.unwrap().is_none());
}

#[tokio::test]
#[serial]
async fn process_header_missing_payload() {
    let (name, secret) = keys().pop().unwrap();
    let signature_service = SignatureService::new(secret);

    let (tx_sync_headers, _rx_sync_headers) = channel(1);
    let (tx_sync_certificates, _rx_sync_certificates) = channel(1);
    let (tx_primary_messages, rx_primary_messages) = channel(1);
    let (_tx_headers_loopback, rx_headers_loopback) = channel(1);
    let (_tx_headers, rx_headers) = channel(1);
    let (tx_parents, _rx_parents) = channel(1);

    let (tx_committer, _rx_committer) = channel(1);
    let (_tx_request_header_sync, rx_request_header_sync) = channel(1);
    let (tx_info, _rx_info) = channel(1);
    let (tx_header_waiter_instances, rx_header_waiter_instances) = channel(1);

    // Create a new test store.
    let path = ".db_test_process_header_missing_payload";
    let _ = fs::remove_dir_all(path);
    let mut store = Store::new(path).unwrap();

    // Make a synchronizer for the core.
    let synchronizer = Synchronizer::new(
        name,
        &committee(),
        store.clone(),
        /* tx_header_waiter */ tx_sync_headers,
        /* tx_certificate_waiter */ tx_sync_certificates,
        false, // use_fast_sync
    );

    let leader_elector = LeaderElector::new(committee().clone());
    let timeout_delay = 1000;

    let parameters = Parameters::default();

    // Spawn the core.
    spawn_test_core(
        name,
        committee(),
        store.clone(),
        synchronizer,
        signature_service,
        /* consensus_round */ Arc::new(AtomicU64::new(0)),
        /* gc_depth */ 50,
        /* rx_primaries */ rx_primary_messages,
        /* rx_header_waiter */ rx_headers_loopback,
        rx_header_waiter_instances,
        /* rx_proposer */ rx_headers,
        tx_committer,
        /* tx_proposer */ tx_parents,
        rx_request_header_sync,
        tx_info,
        leader_elector,
        parameters.timeout_delay,
        parameters.use_optimistic_tips,
        parameters.use_parallel_proposals,
        parameters.k,
        parameters.use_fast_path,
        parameters.fast_path_timeout,
        parameters.use_ride_share,
        parameters.car_timeout,
    );

    // Send a header to the core.
    let header = Header {
        payload: [(Digest::default(), 0)].iter().cloned().collect(),
        ..header()
    };
    let id = header.id.clone();
    tx_primary_messages
        .send(PrimaryMessage::Header(header, false))
        .await
        .unwrap();

    // Ensure the header is not stored.
    assert!(store.read(id.to_vec()).await.unwrap().is_none());
}

#[tokio::test]
#[serial]
async fn process_votes() {
    let (name, secret) = keys().pop().unwrap();
    let signature_service = SignatureService::new(secret);

    let committee = committee_with_base_port(13_100);

    let (tx_sync_headers, _rx_sync_headers) = channel(1);
    let (tx_sync_certificates, _rx_sync_certificates) = channel(1);
    let (tx_primary_messages, rx_primary_messages) = channel(1);
    let (_tx_headers_loopback, rx_headers_loopback) = channel(1);
    let (tx_headers, rx_headers) = channel(1);
    let (tx_parents, mut rx_parents) = channel(1);

    let (tx_committer, _rx_committer) = channel(1);
    let (_tx_request_header_sync, rx_request_header_sync) = channel(1);
    let (tx_info, mut rx_info) = channel(1);
    let (tx_header_waiter_instances, rx_header_waiter_instances) = channel(1);

    // Create a new test store.
    let path = ".db_test_process_vote";
    let _ = fs::remove_dir_all(path);
    let mut store = Store::new(path).unwrap();

    // Make a synchronizer for the core.
    let synchronizer = Synchronizer::new(
        name,
        &committee,
        store.clone(),
        /* tx_header_waiter */ tx_sync_headers,
        /* tx_certificate_waiter */ tx_sync_certificates,
        false, // use_fast_sync
    );

    let leader_elector = LeaderElector::new(committee.clone());
    let timeout_delay = 1000;

    let parameters = Parameters::default();

    // Spawn the core.
    spawn_test_core(
        name,
        committee.clone(),
        store.clone(),
        synchronizer,
        signature_service,
        /* consensus_round */ Arc::new(AtomicU64::new(0)),
        /* gc_depth */ 50,
        /* rx_primaries */ rx_primary_messages,
        /* rx_header_waiter */ rx_headers_loopback,
        rx_header_waiter_instances,
        /* rx_proposer */ rx_headers,
        tx_committer,
        /* tx_proposer */ tx_parents,
        rx_request_header_sync,
        tx_info,
        leader_elector,
        parameters.timeout_delay,
        parameters.use_optimistic_tips,
        parameters.use_parallel_proposals,
        parameters.k,
        parameters.use_fast_path,
        parameters.fast_path_timeout,
        parameters.use_ride_share,
        parameters.car_timeout,
    );

    // Receive geneis parent cert from the proposer
    rx_parents.recv().await.unwrap();

    let header = header();
    // Make the certificate we expect to receive.
    let expected = certificate(&header);

    //Note: core uses Header::genesis instead of Header::default now
    tx_headers.send(header.clone()).await.unwrap();
    sleep(Duration::from_millis(500)).await;
    /*tx_primary_messages
    .send(PrimaryMessage::Header(header.clone()))
    .await
    .unwrap();*/

    // Send a votes to the core.
    for vote in votes(&header) {
        //println!("Vote origin is {:?}", vote.origin);
        tx_primary_messages
            .send(PrimaryMessage::Vote(vote))
            .await
            .unwrap();
    }

    let received_cert = rx_parents.recv().await.unwrap();
    assert_eq!(received_cert.height, expected.height);
    assert_eq!(received_cert.author, expected.author);
    assert_eq!(received_cert.header_digest, expected.header_digest);
    //println!("Expected cert is {:?}, {:?}", expected.header_digest, expected.height);

    // Ensure the listener received the certificate and stored it.
    /*let stored = store
        .read(expected.digest().to_vec())
        .await
        .unwrap()
        .map(|x| bincode::deserialize(&x).unwrap());
    assert_eq!(stored, Some(expected));*/
}

#[tokio::test]
#[serial]
async fn process_certificates() {
    let (name, secret) = keys().pop().unwrap();
    let signature_service = SignatureService::new(secret);

    let (tx_sync_headers, _rx_sync_headers) = channel(1);
    let (tx_sync_certificates, _rx_sync_certificates) = channel(1);
    let (tx_primary_messages, rx_primary_messages) = channel(3);
    let (_tx_headers_loopback, rx_headers_loopback) = channel(1);
    let (tx_headers, rx_headers) = channel(1);
    let (tx_parents, mut rx_parents) = channel(1);

    let (tx_committer, mut rx_committer) = channel(3);
    let (_tx_request_header_sync, rx_request_header_sync) = channel(1);
    let (tx_info, _rx_info) = channel(1);
    let (tx_header_waiter_instances, rx_header_waiter_instances) = channel(1);

    // Create a new test store.
    let path = ".db_test_process_certificates";
    let _ = fs::remove_dir_all(path);
    let mut store = Store::new(path).unwrap();

    // Make a synchronizer for the core.
    let synchronizer = Synchronizer::new(
        name,
        &committee(),
        store.clone(),
        /* tx_header_waiter */ tx_sync_headers,
        /* tx_certificate_waiter */ tx_sync_certificates,
        false, // use_fast_sync
    );

    let leader_elector = LeaderElector::new(committee().clone());
    let timeout_delay = 1000;

    let parameters = Parameters::default();

    // Spawn the core.
    spawn_test_core(
        name,
        committee(),
        store.clone(),
        synchronizer,
        signature_service,
        /* consensus_round */ Arc::new(AtomicU64::new(0)),
        /* gc_depth */ 50,
        /* rx_primaries */ rx_primary_messages,
        /* rx_header_waiter */ rx_headers_loopback,
        rx_header_waiter_instances,
        /* rx_proposer */ rx_headers,
        tx_committer,
        /* tx_proposer */ tx_parents,
        rx_request_header_sync,
        tx_info,
        leader_elector,
        parameters.timeout_delay,
        parameters.use_optimistic_tips,
        parameters.use_parallel_proposals,
        parameters.k,
        parameters.use_fast_path,
        parameters.fast_path_timeout,
        parameters.use_ride_share,
        parameters.car_timeout,
    );

    // Send enough certificates to the core.
    let certificates: Vec<Certificate> =
        headers().iter().map(|header| certificate(header)).collect();

    // Send enough headers to the core.
    let headers_from_certs: Vec<Header> = certificates
        .iter()
        .map(|cert| header_from_cert(cert))
        .collect();

    for x in headers().iter() {
        //println!("author is {:?}", x.author);
        tx_primary_messages
            .send(PrimaryMessage::Header(x.clone(), false))
            .await
            .unwrap();
    }

    for x in headers_from_certs {
        //println!("Sending headers with author {:?}", x.author);
        tx_primary_messages
            .send(PrimaryMessage::Header(x, false))
            .await
            .unwrap();
    }

    // Ensure the certificates are stored.
    for x in &certificates {
        //println!("Cert digest is {:?}", x.digest());
        let stored = store.read(x.digest().to_vec()).await.unwrap();
        let serialized = bincode::serialize(x).unwrap();
        assert_eq!(stored, Some(serialized));
    }
}

#[tokio::test]
#[serial]
async fn process_prepare() {
    let mut keys = keys();
    let _ = keys.pop().unwrap(); // Skip the header' author.
    let (name, secret) = keys.pop().unwrap();
    let mut signature_service = SignatureService::new(secret);

    let committee = committee_with_base_port(13_000);

    let (tx_sync_headers, _rx_sync_headers) = channel(1);
    let (tx_sync_certificates, _rx_sync_certificates) = channel(1);
    let (tx_primary_messages, rx_primary_messages) = channel(1);
    let (_tx_headers_loopback, rx_headers_loopback) = channel(1);
    let (_tx_headers, rx_headers) = channel(1);
    let (tx_parents, _rx_parents) = channel(1);

    let (tx_committer, _rx_committer) = channel(1);
    let (_tx_request_header_sync, rx_request_header_sync) = channel(1);
    let (tx_info, _rx_info) = channel(1);
    let (tx_header_waiter_instances, rx_header_waiter_instances) = channel(1);

    // Create a new test store.
    let path = ".db_test_process_header";
    let _ = fs::remove_dir_all(path);
    let mut store = Store::new(path).unwrap();

    // Spawn a listener to receive the vote.
    let address = committee
        .primary(&header().author)
        .unwrap()
        .primary_to_primary;
    //let handle = listener(address);

    // Make a synchronizer for the core.
    let synchronizer = Synchronizer::new(
        name,
        &committee,
        store.clone(),
        /* tx_header_waiter */ tx_sync_headers,
        /* tx_certificate_waiter */ tx_sync_certificates,
        false, // use_fast_sync
    );

    let leader_elector = LeaderElector::new(committee.clone());
    let timeout_delay = 1000;

    let parameters = Parameters::default();

    // Spawn the core.
    spawn_test_core(
        name,
        committee.clone(),
        store.clone(),
        synchronizer,
        signature_service,
        /* consensus_round */ Arc::new(AtomicU64::new(0)),
        /* gc_depth */ 50,
        /* rx_primaries */ rx_primary_messages,
        /* rx_header_waiter */ rx_headers_loopback,
        rx_header_waiter_instances,
        /* rx_proposer */ rx_headers,
        tx_committer,
        /* tx_proposer */ tx_parents,
        rx_request_header_sync,
        tx_info,
        leader_elector,
        parameters.timeout_delay,
        parameters.use_optimistic_tips,
        parameters.use_parallel_proposals,
        parameters.k,
        parameters.use_fast_path,
        parameters.fast_path_timeout,
        parameters.use_ride_share,
        parameters.car_timeout,
    );

    // Send headers to the core, so they won't request sync
    let header_list = headers();
    for x in header_list.clone() {
        tx_primary_messages
            .send(PrimaryMessage::Header(x, false))
            .await
            .unwrap();
    }

    let mut proposals: HashMap<PublicKey, Proposal> = HashMap::new();
    for x in &header_list {
        proposals.insert(
            x.author,
            Proposal {
                header_digest: x.digest(),
                height: x.height(),
            },
        );
    }
    let prepare_message: ConsensusMessage = ConsensusMessage::Prepare {
        slot: 1,
        view: 1,
        tc: None,
        qc_ticket: None,
        proposals,
        aggregate_report: None,
    };

    let mut consensus_messages: HashMap<Digest, ConsensusMessage> = HashMap::new();
    consensus_messages.insert(prepare_message.digest(), prepare_message.clone());

    let parent_cert = certificate(&header_list[0]);
    let header = special_header(parent_cert, consensus_messages);

    // Send a header to the core.
    tx_primary_messages
        .send(PrimaryMessage::Header(header.clone(), false))
        .await
        .unwrap();

    listener(address).await.unwrap();

    // Make the vote we expect to receive.
    let handle = listener(address);
    let received = handle.await.unwrap();
    match bincode::deserialize(&received).unwrap() {
        PrimaryMessage::Vote(x) => {
            //assert_eq!(x, expected);
            assert!(!x.consensus_votes.is_empty());
            assert_eq!(x.height, 2);
            assert_eq!(prepare_message.digest(), x.consensus_votes[0].1);
        }
        x => panic!("Unexpected message: {:?}", x),
    }

    // Ensure the header is correctly stored.
    let stored = store
        .read(header.id.to_vec())
        .await
        .unwrap()
        .map(|x| bincode::deserialize(&x).unwrap());
    assert_eq!(stored, Some(header));
}

#[tokio::test]
#[serial]
async fn generate_confirm() {
    let mut keys = keys();
    let _ = keys.pop().unwrap(); // Skip the header' author.
    let (name, secret) = keys.pop().unwrap();
    let mut signature_service = SignatureService::new(secret);

    let committee = committee_with_base_port(13_000);

    let (tx_sync_headers, _rx_sync_headers) = channel(1);
    let (tx_sync_certificates, _rx_sync_certificates) = channel(1);
    let (tx_primary_messages, rx_primary_messages) = channel(1);
    let (_tx_headers_loopback, rx_headers_loopback) = channel(1);
    let (tx_headers, rx_headers) = channel(1);
    let (tx_parents, mut rx_parents) = channel(1);

    let (tx_committer, _rx_committer) = channel(1);
    let (_tx_request_header_sync, rx_request_header_sync) = channel(1);
    let (tx_info, mut rx_info) = channel(1);
    let (tx_header_waiter_instances, rx_header_waiter_instances) = channel(1);

    // Create a new test store.
    let path = ".db_test_process_header";
    let _ = fs::remove_dir_all(path);
    let mut store = Store::new(path).unwrap();

    // Spawn a listener to receive the vote.
    let address = committee
        .primary(&header().author)
        .unwrap()
        .primary_to_primary;
    //let handle = listener(address);

    // Make a synchronizer for the core.
    let synchronizer = Synchronizer::new(
        name,
        &committee,
        store.clone(),
        /* tx_header_waiter */ tx_sync_headers,
        /* tx_certificate_waiter */ tx_sync_certificates,
        false, // use_fast_sync
    );

    let leader_elector = LeaderElector::new(committee.clone());
    let timeout_delay = 1000;

    let parameters = Parameters::default();

    // Spawn the core.
    spawn_test_core(
        name,
        committee.clone(),
        store.clone(),
        synchronizer,
        signature_service,
        /* consensus_round */ Arc::new(AtomicU64::new(0)),
        /* gc_depth */ 50,
        /* rx_primaries */ rx_primary_messages,
        /* rx_header_waiter */ rx_headers_loopback,
        rx_header_waiter_instances,
        /* rx_proposer */ rx_headers,
        tx_committer,
        /* tx_proposer */ tx_parents,
        rx_request_header_sync,
        tx_info,
        leader_elector,
        parameters.timeout_delay,
        parameters.use_optimistic_tips,
        parameters.use_parallel_proposals,
        parameters.k,
        parameters.use_fast_path,
        parameters.fast_path_timeout,
        parameters.use_ride_share,
        parameters.car_timeout,
    );

    // Receive the first prepare message from proposer
    rx_info.recv().await.unwrap();
    rx_parents.recv().await.unwrap();

    /*Proposer::spawn(
        name,
        committee.clone(),
        signature_service,
        100,
        timeout_delay,
        rx_parents,
        rx_workers,
        rx_info,
        tx_headers,
    );*/

    // Send headers to the core, so they won't request sync
    let header_list = headers();
    for x in header_list.clone() {
        tx_primary_messages
            .send(PrimaryMessage::Header(x, false))
            .await
            .unwrap();
    }

    let mut proposals: HashMap<PublicKey, Proposal> = HashMap::new();
    for x in &header_list {
        proposals.insert(
            x.author,
            Proposal {
                header_digest: x.digest(),
                height: x.height(),
            },
        );
    }
    let prepare_message: ConsensusMessage = ConsensusMessage::Prepare {
        slot: 1,
        view: 1,
        tc: None,
        qc_ticket: None,
        proposals: proposals.clone(),
        aggregate_report: None,
    };

    let mut consensus_messages: HashMap<Digest, ConsensusMessage> = HashMap::new();
    consensus_messages.insert(prepare_message.digest().clone(), prepare_message.clone());

    let parent_cert = certificate(&header_list[0]);
    let header = special_header(parent_cert, consensus_messages.clone());
    let consensus_digests = vec![prepare_message.digest()];

    // Send a header to the core.
    tx_headers.send(header.clone()).await.unwrap();

    for vote in special_votes(&header, consensus_digests) {
        //println!("sending special votes");
        let message = PrimaryMessage::Vote(vote);
        tx_primary_messages.send(message).await.unwrap();
    }

    let confirm_message = rx_info.recv().await.unwrap();
    match confirm_message {
        ConsensusMessage::Confirm {
            slot,
            view,
            qc,
            proposals: _,
            aggregate_report: _,
        } => {
            assert_eq!(slot, 1);
            assert_eq!(view, 1);
            assert_eq!(qc.id, prepare_message.digest().clone());
            assert_eq!(qc.votes.len(), 3);
        }
        _ => panic!("Wrong message type"),
    };
}

#[tokio::test]
#[serial]
async fn generate_commit() {
    let mut keys = keys();
    let _ = keys.pop().unwrap(); // Skip the header' author.
    let (name, secret) = keys.pop().unwrap();
    let mut signature_service = SignatureService::new(secret);

    let committee = committee_with_base_port(13_000);

    let (tx_sync_headers, _rx_sync_headers) = channel(1);
    let (tx_sync_certificates, _rx_sync_certificates) = channel(1);
    let (tx_primary_messages, rx_primary_messages) = channel(1);
    let (_tx_headers_loopback, rx_headers_loopback) = channel(1);
    let (tx_headers, mut rx_headers) = channel(1);
    let (tx_parents, mut rx_parents) = channel(1);

    let (tx_committer, mut rx_committer) = channel(1);
    let (_tx_request_header_sync, rx_request_header_sync) = channel(1);
    let (tx_info, mut rx_info) = channel(1);
    let (tx_header_waiter_instances, rx_header_waiter_instances) = channel(1);

    // Create a new test store.
    let path = ".db_test_process_header";
    let _ = fs::remove_dir_all(path);
    let mut store = Store::new(path).unwrap();

    // Spawn a listener to receive the vote.
    let address = committee
        .primary(&header().author)
        .unwrap()
        .primary_to_primary;
    //let handle = listener(address);

    // Make a synchronizer for the core.
    let synchronizer = Synchronizer::new(
        name,
        &committee,
        store.clone(),
        /* tx_header_waiter */ tx_sync_headers,
        /* tx_certificate_waiter */ tx_sync_certificates,
        false, // use_fast_sync
    );

    let leader_elector = LeaderElector::new(committee.clone());
    let timeout_delay = 1000;

    let parameters = Parameters::default();

    // Spawn the core.
    spawn_test_core(
        name,
        committee.clone(),
        store.clone(),
        synchronizer,
        signature_service,
        /* consensus_round */ Arc::new(AtomicU64::new(0)),
        /* gc_depth */ 50,
        /* rx_primaries */ rx_primary_messages,
        /* rx_header_waiter */ rx_headers_loopback,
        rx_header_waiter_instances,
        /* rx_proposer */ rx_headers,
        tx_committer,
        /* tx_proposer */ tx_parents,
        rx_request_header_sync,
        tx_info,
        leader_elector,
        parameters.timeout_delay,
        parameters.use_optimistic_tips,
        parameters.use_parallel_proposals,
        parameters.k,
        parameters.use_fast_path,
        parameters.fast_path_timeout,
        parameters.use_ride_share,
        parameters.car_timeout,
    );

    // Receive the first prepare message from proposer
    rx_info.recv().await.unwrap();
    rx_parents.recv().await.unwrap();

    /*Proposer::spawn(
        name,
        committee.clone(),
        signature_service,
        100,
        timeout_delay,
        rx_parents,
        rx_workers,
        rx_info,
        tx_headers,
    );*/

    // Send headers to the core, so they won't request sync
    let header_list = headers();
    for x in header_list.clone() {
        tx_primary_messages
            .send(PrimaryMessage::Header(x, false))
            .await
            .unwrap();
    }

    let mut proposals: HashMap<PublicKey, Proposal> = HashMap::new();
    for x in &header_list {
        proposals.insert(
            x.author,
            Proposal {
                header_digest: x.digest(),
                height: x.height(),
            },
        );
    }
    let prepare_message: ConsensusMessage = ConsensusMessage::Prepare {
        slot: 1,
        view: 1,
        tc: None,
        qc_ticket: None,
        proposals: proposals.clone(),
        aggregate_report: None,
    };

    let mut consensus_messages: HashMap<Digest, ConsensusMessage> = HashMap::new();
    consensus_messages.insert(prepare_message.digest().clone(), prepare_message.clone());

    let parent_cert = certificate(&header_list[0]);
    let header = special_header(parent_cert, consensus_messages.clone());
    let consensus_digests = vec![prepare_message.digest()];

    // Send a header to the core.
    tx_headers.send(header.clone()).await.unwrap();

    for vote in special_votes(&header, consensus_digests) {
        //println!("sending special votes");
        let message = PrimaryMessage::Vote(vote);
        tx_primary_messages.send(message).await.unwrap();
    }

    let confirm_message = rx_info.recv().await.unwrap();
    match confirm_message.clone() {
        ConsensusMessage::Confirm {
            slot,
            view,
            qc,
            proposals: _,
            aggregate_report: _,
        } => {
            consensus_messages.clear();
            consensus_messages.insert(confirm_message.digest().clone(), confirm_message.clone());

            let confirm_parent_cert = certificate(&header);
            let confirm_header = special_header(confirm_parent_cert, consensus_messages.clone());

            //println!("confirm header height {:?} author {:?}", confirm_header.height, confirm_header.author);

            tx_headers.send(confirm_header.clone()).await.unwrap();

            sleep(Duration::from_millis(500)).await;
            let confirm_digests = vec![confirm_message.digest().clone()];

            for vote in special_votes(&confirm_header, confirm_digests) {
                //println!("sending special votes confirm");
                let message = PrimaryMessage::Vote(vote);
                tx_primary_messages.send(message).await.unwrap();
            }

            //println!("after sending votes");

            let commit_message = rx_info.recv().await.unwrap();
            rx_parents.recv().await.unwrap();

            match commit_message.clone() {
                ConsensusMessage::Commit {
                    slot: slot1,
                    view: view1,
                    qc: qc1,
                    proposals: proposals1,
                    aggregate_report: aggregate_report1,
                } => {
                    consensus_messages.clear();
                    consensus_messages
                        .insert(commit_message.digest().clone(), commit_message.clone());

                    let commit_parent_cert = certificate(&confirm_header);
                    let commit_header =
                        special_header(commit_parent_cert, consensus_messages.clone());

                    //println!("sending commit header: {:?}, {:?}", commit_header.height, commit_header.author);

                    // Ensure that the confirm header is processed and received
                    //sleep(Duration::from_millis(500)).await;
                    tx_headers.send(commit_header).await.unwrap();

                    //println!("awaiting committer");
                    let (receive_commit_message, _fast_commit) = rx_committer.recv().await.unwrap();
                    match receive_commit_message {
                        ConsensusMessage::Commit {
                            slot: slot2,
                            view: view2,
                            qc: qc2,
                            proposals: proposals2,
                            aggregate_report: aggregate_report2,
                        } => {
                            assert_eq!(slot1, slot2);
                            assert_eq!(view1, view2);
                            assert_eq!(
                                aggregate_report1.as_ref().map(|r| r.digest()),
                                aggregate_report2.as_ref().map(|r| r.digest())
                            );
                        }
                        _ => {}
                    };
                }
                _ => {}
            };
        }
        _ => panic!("Wrong message type"),
    };
}

#[tokio::test]
#[serial]
async fn generate_pipelined_prepare() {
    let mut keys = keys();
    let _ = keys.pop().unwrap(); // Skip the header' author.
    let (name, secret) = keys.pop().unwrap();
    let mut signature_service = SignatureService::new(secret);

    let committee = committee_with_base_port(13_000);

    let (tx_sync_headers, _rx_sync_headers) = channel(1);
    let (tx_sync_certificates, _rx_sync_certificates) = channel(1);
    let (tx_primary_messages, rx_primary_messages) = channel(1);
    let (_tx_headers_loopback, rx_headers_loopback) = channel(1);
    let (_tx_headers, rx_headers) = channel(1);
    let (tx_parents, mut rx_parents) = channel(1);

    let (tx_committer, _rx_committer) = channel(1);
    let (_tx_request_header_sync, rx_request_header_sync) = channel(1);
    let (tx_info, mut rx_info) = channel(1);
    let (tx_header_waiter_instances, rx_header_waiter_instances) = channel(1);

    // Create a new test store.
    let path = ".db_test_process_header";
    let _ = fs::remove_dir_all(path);
    let mut store = Store::new(path).unwrap();

    // Spawn a listener to receive the vote.
    let address = committee
        .primary(&header().author)
        .unwrap()
        .primary_to_primary;
    //let handle = listener(address);

    // Make a synchronizer for the core.
    let synchronizer = Synchronizer::new(
        name,
        &committee,
        store.clone(),
        /* tx_header_waiter */ tx_sync_headers,
        /* tx_certificate_waiter */ tx_sync_certificates,
        false, // use_fast_sync
    );

    let leader_elector = LeaderElector::new(committee.clone());
    let timeout_delay = 1000;

    let parameters = Parameters::default();

    // Spawn the core.
    spawn_test_core(
        name,
        committee.clone(),
        store.clone(),
        synchronizer,
        signature_service,
        /* consensus_round */ Arc::new(AtomicU64::new(0)),
        /* gc_depth */ 50,
        /* rx_primaries */ rx_primary_messages,
        /* rx_header_waiter */ rx_headers_loopback,
        rx_header_waiter_instances,
        /* rx_proposer */ rx_headers,
        tx_committer,
        /* tx_proposer */ tx_parents,
        rx_request_header_sync,
        tx_info,
        leader_elector,
        parameters.timeout_delay,
        parameters.use_optimistic_tips,
        parameters.use_parallel_proposals,
        parameters.k,
        parameters.use_fast_path,
        parameters.fast_path_timeout,
        parameters.use_ride_share,
        parameters.car_timeout,
    );

    // Receive the first prepare message from proposer
    rx_info.recv().await.unwrap();
    rx_parents.recv().await.unwrap();

    // Send headers to the core, so they won't request sync
    let header_list = headers();
    for x in header_list.clone() {
        tx_primary_messages
            .send(PrimaryMessage::Header(x, false))
            .await
            .unwrap();
    }

    let mut proposals: HashMap<PublicKey, Proposal> = HashMap::new();
    for x in &header_list {
        proposals.insert(
            x.author,
            Proposal {
                header_digest: x.digest(),
                height: x.height(),
            },
        );
    }
    let prepare_message: ConsensusMessage = ConsensusMessage::Prepare {
        slot: 1,
        view: 1,
        tc: None,
        qc_ticket: None,
        proposals,
        aggregate_report: None,
    };

    let mut consensus_messages: HashMap<Digest, ConsensusMessage> = HashMap::new();
    consensus_messages.insert(prepare_message.digest(), prepare_message.clone());

    let parent_cert = certificate(&header_list[0]);
    let header = special_header(parent_cert, consensus_messages);

    // Send a header to the core.
    tx_primary_messages
        .send(PrimaryMessage::Header(header.clone(), false))
        .await
        .unwrap();

    // Send enough certificates to the core.
    let certificates: Vec<Certificate> = headers()
        .iter()
        .rev()
        .skip(1)
        .map(|header| certificate(header))
        .collect();

    // Send enough headers to the core.
    let headers_from_certs: Vec<Header> = certificates
        .iter()
        .rev()
        .skip(1)
        .map(|cert| header_from_cert(cert))
        .collect();

    for x in headers_from_certs.clone() {
        //println!("header author is {:?}", x.author);
        tx_primary_messages
            .send(PrimaryMessage::Header(x, false))
            .await
            .unwrap();
    }

    // Send a header to the core.
    //println!("special header author is {:?}", header.author);
    tx_primary_messages
        .send(PrimaryMessage::Header(header.clone(), false))
        .await
        .unwrap();

    listener(address).await.unwrap();
    let output_message = rx_info.recv().await.unwrap();

    match output_message {
        ConsensusMessage::Prepare {
            slot,
            view,
            tc: _,
            qc_ticket: _,
            proposals: _,
            aggregate_report: _,
        } => {
            assert_eq!(slot, 2);
            assert_eq!(view, 1);
        }
        _ => {}
    };

    // Ensure the header is correctly stored.
    let stored = store
        .read(header.id.to_vec())
        .await
        .unwrap()
        .map(|x| bincode::deserialize(&x).unwrap());
    assert_eq!(stored, Some(header));
}

#[tokio::test]
#[serial]
async fn local_timeout_view() {
    let mut keys = keys();
    let _ = keys.pop().unwrap(); // Skip the header' author.
    let (name, secret) = keys.pop().unwrap();
    let mut signature_service = SignatureService::new(secret);

    let committee = committee_with_base_port(13_000);

    let (tx_sync_headers, _rx_sync_headers) = channel(1);
    let (tx_sync_certificates, _rx_sync_certificates) = channel(1);
    let (tx_primary_messages, rx_primary_messages) = channel(1);
    let (_tx_headers_loopback, rx_headers_loopback) = channel(1);
    let (_tx_headers, rx_headers) = channel(1);
    let (tx_parents, _rx_parents) = channel(1);

    let (tx_committer, _rx_committer) = channel(1);
    let (_tx_request_header_sync, rx_request_header_sync) = channel(1);
    let (tx_info, mut rx_info) = channel(1);
    let (tx_header_waiter_instances, rx_header_waiter_instances) = channel(1);

    // Create a new test store.
    let path = ".db_test_process_header";
    let _ = fs::remove_dir_all(path);
    let mut store = Store::new(path).unwrap();

    // Spawn a listener to receive the vote.
    let address = committee
        .primary(&header().author)
        .unwrap()
        .primary_to_primary;
    let handle = listener(address);

    // Make a synchronizer for the core.
    let synchronizer = Synchronizer::new(
        name,
        &committee,
        store.clone(),
        /* tx_header_waiter */ tx_sync_headers,
        /* tx_certificate_waiter */ tx_sync_certificates,
        false, // use_fast_sync
    );

    let leader_elector = LeaderElector::new(committee.clone());
    let timeout_delay = 1000;

    let parameters = Parameters::default();

    // Spawn the core.
    spawn_test_core(
        name,
        committee.clone(),
        store.clone(),
        synchronizer,
        signature_service,
        /* consensus_round */ Arc::new(AtomicU64::new(0)),
        /* gc_depth */ 50,
        /* rx_primaries */ rx_primary_messages,
        /* rx_header_waiter */ rx_headers_loopback,
        rx_header_waiter_instances,
        /* rx_proposer */ rx_headers,
        tx_committer,
        /* tx_proposer */ tx_parents,
        rx_request_header_sync,
        tx_info,
        leader_elector,
        parameters.timeout_delay,
        parameters.use_optimistic_tips,
        parameters.use_parallel_proposals,
        parameters.k,
        parameters.use_fast_path,
        parameters.fast_path_timeout,
        parameters.use_ride_share,
        parameters.car_timeout,
    );

    /*let message = handle.await.unwrap();

    match bincode::deserialize(&message).unwrap() {
         PrimaryMessage::Timeout(timeout) => {
             assert_eq!(timeout.slot, 1);
             assert_eq!(timeout.view, 1);
         }
         x => panic!("Unexpected message: {:?}", x),
     };*/
}

#[tokio::test]
#[serial]
async fn sync_missing_proposals() {
    let mut keys = keys();
    let _ = keys.pop().unwrap(); // Skip the header' author.
    let (name, secret) = keys.pop().unwrap();
    let mut signature_service = SignatureService::new(secret);

    let committee = committee_with_base_port(13_000);

    let (tx_sync_headers, rx_sync_headers) = channel(1);
    let (tx_sync_certificates, _rx_sync_certificates) = channel(1);
    let (tx_primary_messages, rx_primary_messages) = channel(1);
    let (tx_headers_loopback, rx_headers_loopback) = channel(1);
    let (_tx_headers, rx_headers) = channel(1);
    let (tx_parents, mut rx_parents) = channel(1);

    let (tx_committer, _rx_committer) = channel(1);
    let (_tx_request_header_sync, rx_request_header_sync) = channel(1);
    let (tx_info, mut rx_info) = channel(1);
    let (tx_header_waiter_instances, rx_header_waiter_instances) = channel(1);

    // Create a new test store.
    let path = ".db_test_process_header";
    let _ = fs::remove_dir_all(path);
    let mut store = Store::new(path).unwrap();

    // Spawn a listener to receive the vote.
    let address = committee
        .primary(&header().author)
        .unwrap()
        .primary_to_primary;
    //let handle = listener(address);

    // Make a synchronizer for the core.
    let synchronizer = Synchronizer::new(
        name,
        &committee,
        store.clone(),
        /* tx_header_waiter */ tx_sync_headers,
        /* tx_certificate_waiter */ tx_sync_certificates,
        false, // use_fast_sync
    );

    let leader_elector = LeaderElector::new(committee.clone());
    let timeout_delay = 100000;

    let parameters = Parameters::default();

    // Spawn the core.
    spawn_test_core(
        name,
        committee.clone(),
        store.clone(),
        synchronizer,
        signature_service,
        /* consensus_round */ Arc::new(AtomicU64::new(0)),
        /* gc_depth */ 50,
        /* rx_primaries */ rx_primary_messages,
        /* rx_header_waiter */ rx_headers_loopback,
        rx_header_waiter_instances,
        /* rx_proposer */ rx_headers,
        tx_committer,
        /* tx_proposer */ tx_parents,
        rx_request_header_sync,
        tx_info,
        leader_elector,
        parameters.timeout_delay,
        parameters.use_optimistic_tips,
        parameters.use_parallel_proposals,
        parameters.k,
        parameters.use_fast_path,
        parameters.fast_path_timeout,
        parameters.use_ride_share,
        parameters.car_timeout,
    );

    // Receive the first prepare message from proposer
    rx_info.recv().await.unwrap();
    rx_parents.recv().await.unwrap();

    HeaderWaiter::spawn(
        name,
        committee,
        store.clone(),
        Arc::new(AtomicU64::new(0)),
        50,
        timeout_delay,
        1,
        rx_sync_headers,
        tx_headers_loopback,
        tx_header_waiter_instances,
        parameters.use_fast_sync,
        parameters.use_optimistic_tips,
    );

    // Send headers to the core, so they won't request sync
    let header_list = headers();
    /*for x in header_list.clone() {
        tx_primary_messages
            .send(PrimaryMessage::Header(x))
            .await
            .unwrap();
    }*/

    tx_primary_messages
        .send(PrimaryMessage::Header(header_list[0].clone(), false))
        .await
        .unwrap();

    let mut proposals: HashMap<PublicKey, Proposal> = HashMap::new();
    for x in &header_list {
        proposals.insert(
            x.author,
            Proposal {
                header_digest: x.digest(),
                height: x.height(),
            },
        );
    }
    let prepare_message: ConsensusMessage = ConsensusMessage::Prepare {
        slot: 1,
        view: 1,
        tc: None,
        qc_ticket: None,
        proposals,
        aggregate_report: None,
    };

    let mut consensus_messages: HashMap<Digest, ConsensusMessage> = HashMap::new();
    consensus_messages.insert(prepare_message.digest(), prepare_message.clone());

    let parent_cert = certificate(&header_list[0]);
    let header = special_header(parent_cert, consensus_messages);

    // Send the special header to the core, should trigger sync
    tx_primary_messages
        .send(PrimaryMessage::Header(header.clone(), false))
        .await
        .unwrap();

    sleep(Duration::from_millis(500)).await;
    // Send the misssing proposals
    for x in header_list.into_iter().skip(1) {
        //println!("header author is {:?}", x.author);
        tx_primary_messages
            .send(PrimaryMessage::Header(x, false))
            .await
            .unwrap();
    }

    // Wait for the proposals to appear in the store
    sleep(Duration::from_millis(500)).await;

    // Ensure the header is correctly stored.
    let stored = store
        .read(header.id.to_vec())
        .await
        .unwrap()
        .map(|x| bincode::deserialize(&x).unwrap());
    assert_eq!(stored, Some(header));
}

/*#[tokio::test]
#[serial]
async fn process_special_header() {
    let mut keys = keys();
    let _ = keys.pop().unwrap(); // Skip the header' author.
    let (name, secret) = keys.pop().unwrap();
    let mut signature_service = SignatureService::new(secret);

    let committee = committee_with_base_port(13_000);

    let (tx_sync_headers, _rx_sync_headers) = channel(1);
    let (tx_sync_certificates, _rx_sync_certificates) = channel(1);
    let (tx_primary_messages, rx_primary_messages) = channel(1);
    let (_tx_headers_loopback, rx_headers_loopback) = channel(1);
    let (_tx_certificates_loopback, rx_certificates_loopback) = channel(1);
    let (_tx_headers, rx_headers) = channel(1);
    let (tx_consensus, _rx_consensus) = channel(1);
    let (tx_parents, _rx_parents) = channel(1);

    let(tx_committer, _rx_committer) = channel(1);
    let(tx_validation, rx_validation) = channel(1);
    let(tx_sailfish, mut rx_special) = channel(1);
    let(_tx_pushdown_cert, rx_pushdown_cert) = channel(1);
    let(_tx_request_header_sync, rx_request_header_sync) = channel(1);

    // Create a new test store.
    let path = ".db_test_process_header";
    let _ = fs::remove_dir_all(path);
    let mut store = Store::new(path).unwrap();

    // Make the vote we expect to receive.
    //let expected = Vote::new(&header(), &name, &mut signature_service, 0u8, None, None).await;
    let special_expected = Vote::new(&special_header(), &name, &mut signature_service, true).await;

    // Spawn a listener to receive the vote.
    let address = committee
        .primary(&header().author)
        .unwrap()
        .primary_to_primary;
    let handle = listener(address.clone());


    // Make a synchronizer for the core.
    let synchronizer = Synchronizer::new(
        name,
        &committee,
        store.clone(),
        /* tx_header_waiter */ tx_sync_headers,
        /* tx_certificate_waiter */ tx_sync_certificates,
    );

    // Spawn the core.
    Core::spawn(
        name,
        committee,
        store.clone(),
        synchronizer,
        signature_service,
        /* consensus_round */ Arc::new(AtomicU64::new(0)),
        /* gc_depth */ 50,
        /* rx_primaries */ rx_primary_messages,
        /* rx_header_waiter */ rx_headers_loopback,
        /* rx_certificate_waiter */ rx_certificates_loopback,
        /* rx_proposer */ rx_headers,
        tx_consensus,
        tx_committer,
        /* tx_proposer */ tx_parents,
        rx_validation,
        tx_sailfish,
        rx_pushdown_cert,
        rx_request_header_sync
    );

//  // Send a normal header to the core.
//     tx_primary_messages
//         .send(PrimaryMessage::Header(header()))
//         .await
//         .unwrap();


//     //Generates vote:
//     // Ensure the listener correctly received the vote.
//     let received = handle.await.unwrap();
//     match bincode::deserialize(&received).unwrap() {
//         PrimaryMessage::Vote(x) => assert_eq!(x, expected),
//         x => panic!("Unexpected message: {:?}", x),
//     }

//     // Ensure the header is correctly stored.
//     let stored = store
//         .read(header().id.to_vec())
//         .await
//         .unwrap()
//         .map(|x| bincode::deserialize(&x).unwrap());
//     assert_eq!(stored, Some(header()));


//     //// Start special header
//     ////////// once we confirm parent is stored.
//     let handle = listener(address.clone());

    //Send special header with special edge = previous header
    tx_primary_messages
        .send(PrimaryMessage::Header(special_header()))
        .await
        .unwrap();


    //TODO: create receiver for validation
    //send back val result = correct
    let val = rx_special.recv().await.unwrap();
    tx_validation.send((val, 1u8, None, None)).await.unwrap();


    // Ensure the listener correctly received the vote.

    let received = handle.await.unwrap();
    match bincode::deserialize(&received).unwrap() {
        PrimaryMessage::Vote(x) => assert_eq!(x, special_expected),
        x => panic!("Unexpected message: {:?}", x),
    }

    // Ensure the header is correctly stored.
    let stored = store
        .read(special_header().id.to_vec())
        .await
        .unwrap()
        .map(|x| bincode::deserialize(&x).unwrap());
    assert_eq!(stored, Some(special_header()));
}*/

//todo: process special vote

/*#[tokio::test]
#[serial]
async fn process_special_votes() {
    let (name, secret) = keys().pop().unwrap();
    let signature_service = SignatureService::new(secret);

    let committee = committee_with_base_port(13_100);

    let (tx_sync_headers, _rx_sync_headers) = channel(1);
    let (tx_sync_certificates, _rx_sync_certificates) = channel(1);
    let (tx_primary_messages, rx_primary_messages) = channel(1);
    let (_tx_headers_loopback, rx_headers_loopback) = channel(1);
    let (_tx_certificates_loopback, rx_certificates_loopback) = channel(1);
    let (tx_headers, rx_headers) = channel(1);
    let (tx_consensus, _rx_consensus) = channel(1);
    let (tx_parents, _rx_parents) = channel(1);

    let(tx_committer, _rx_committer) = channel(1);
    let(tx_validation, rx_validation) = channel(1);
    let(tx_sailfish, mut rx_special) = channel(1);
    let(_tx_pushdown_cert, rx_pushdown_cert) = channel(1);
    let(_tx_request_header_sync, rx_request_header_sync) = channel(1);

    // Create a new test store.
    let path = ".db_test_process_vote";
    let _ = fs::remove_dir_all(path);
    let store = Store::new(path).unwrap();

    // Make a synchronizer for the core.
    let synchronizer = Synchronizer::new(
        name,
        &committee,
        store.clone(),
        /* tx_header_waiter */ tx_sync_headers,
        /* tx_certificate_waiter */ tx_sync_certificates,
    );

    // Spawn the core.
    Core::spawn(
        name,
        committee.clone(),
        store.clone(),
        synchronizer,
        signature_service,
        /* consensus_round */ Arc::new(AtomicU64::new(0)),
        /* gc_depth */ 50,
        /* rx_primaries */ rx_primary_messages,
        /* rx_header_waiter */ rx_headers_loopback,
        /* rx_certificate_waiter */ rx_certificates_loopback,
        /* rx_proposer */ rx_headers,
        tx_consensus,
        tx_committer,
        /* tx_proposer */ tx_parents,
        rx_validation,
        tx_sailfish,
        rx_pushdown_cert,
        rx_request_header_sync
    );

    // Spawn all listeners to receive our newly formed certificate.
    let vote_handles: Vec<_> = committee
        .others_primaries(&name)
        .iter()
        .map(|(_, address)| listener(address.clone().primary_to_primary))
        .collect();

    //Send new special header to core (to propose itself)

    let header = special_header();
    tx_headers
        .send(header)
        .await
        .unwrap();
    //will broadcast header and process own vote

    //Ensure special header is processed first and becomes "current_header"
    sleep(Duration::from_millis(100)).await;


      // Ensure all listeners got the header.
      for received in try_join_all(vote_handles).await.unwrap() {
        match bincode::deserialize(&received).unwrap() {
            PrimaryMessage::Header(x) => assert_eq!(x, special_header()),
            x => panic!("Unexpected message: {:?}", x),
        }
    }
    //println!("received all votes");
    // // Send a votes to the core. ==> Sending only 2 votes. Supplementing quorum with own vote (called as result of process_header).
    let mut count = 0;
    for vote in special_votes(&special_header()) {
        if vote.author == name {continue;}
        tx_primary_messages
            .send(PrimaryMessage::Vote(vote))
            .await
            .unwrap();
        count = count+1;
        if count == 2 { break;}
    }

    //println!("count {}", count);

     //send back val result = correct ==> allows us to form our own vote.
     let val = rx_special.recv().await.unwrap();
     tx_validation.send((val, name.0[0] % 2, None, None)).await.unwrap();

    //Upon receiving all votes, will broadcast cert.

    // Make the certificate we expect to receive.
    let expected = special_certificate(&special_header());

    // let received = rx_committer.recv().await.unwrap();
    // assert_eq!(received, expected);

    //println!("expected num votes: {}", expected.votes.len());

     // Spawn all listeners to receive our newly formed certificate.
    let cert_handles: Vec<_> = committee
    .others_primaries(&name)
    .iter()
    .map(|(_, address)| listener(address.clone().primary_to_primary))
    .collect();

    // Ensure all listeners got the certificate.
    for received in try_join_all(cert_handles).await.unwrap() {
        match bincode::deserialize(&received).unwrap() {
            PrimaryMessage::Certificate(x) => {//println!{"received cert with {} votes", x.votes.len()}; assert_eq!(x, expected)},
            x => panic!("Unexpected message: {:?}", x),
        }
    }
}*/

/*#[tokio::test]
#[serial]
async fn process_special_certificate() {
    let (name, secret) = keys().pop().unwrap();
    let signature_service = SignatureService::new(secret);

    let (tx_sync_headers, _rx_sync_headers) = channel(1);
    let (tx_sync_certificates, _rx_sync_certificates) = channel(1);
    let (tx_primary_messages, rx_primary_messages) = channel(3);
    let (_tx_headers_loopback, rx_headers_loopback) = channel(1);
    let (_tx_certificates_loopback, rx_certificates_loopback) = channel(1);
    let (_tx_headers, rx_headers) = channel(1);
    let (tx_consensus, mut _rx_consensus) = channel(3);
    let (tx_parents, mut _rx_parents) = channel(1);

    let(tx_committer, mut rx_committer) = channel(2);
    let(_tx_validation, rx_validation) = channel(1);
    let(tx_sailfish, _rx_special) = channel(3);
    let(_tx_pushdown_cert, rx_pushdown_cert) = channel(1);
    let(_tx_request_header_sync, rx_request_header_sync) = channel(1);

    // Create a new test store.
    let path = ".db_test_process_certificates";
    let _ = fs::remove_dir_all(path);
    let mut store = Store::new(path).unwrap();

    // Make a synchronizer for the core.
    let synchronizer = Synchronizer::new(
        name,
        &committee(),
        store.clone(),
        /* tx_header_waiter */ tx_sync_headers,
        /* tx_certificate_waiter */ tx_sync_certificates,
    );

    // Spawn the core.
    Core::spawn(
        name,
        committee(),
        store.clone(),
        synchronizer,
        signature_service,
        /* consensus_round */ Arc::new(AtomicU64::new(0)),
        /* gc_depth */ 50,
        /* rx_primaries */ rx_primary_messages,
        /* rx_header_waiter */ rx_headers_loopback,
        /* rx_certificate_waiter */ rx_certificates_loopback,
        /* rx_proposer */ rx_headers,
        tx_consensus,
        tx_committer,
        /* tx_proposer */ tx_parents,
        rx_validation,
        tx_sailfish,
        rx_pushdown_cert,
        rx_request_header_sync
    );

    //Send one special cert to core
    let cert = special_certificate(&special_header());
    tx_primary_messages
            .send(PrimaryMessage::Certificate(cert.clone()))
            .await
            .unwrap();

    //Make sure cert is stored
    //Make sure committer receives two certs, one for parent, one for self.

    // Ensure the core sends the certificates to the consensus committer.
    let expected_parent_cert = Certificate {
        header_digest: Header::genesis(&committee()).,
        ..Certificate::default()
    };
    let parent = rx_committer.recv().await.unwrap(); //TODO: Receive special parent
    assert_eq!(parent, expected_parent_cert);

    let received = rx_committer.recv().await.unwrap();
    assert_eq!(received, cert);

    // Ensure the certificates are stored.
    let stored = store.read(cert.digest().to_vec()).await.unwrap();
    let serialized = bincode::serialize(&cert).unwrap();
    assert_eq!(stored, Some(serialized));

}*/
