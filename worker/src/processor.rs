// Copyright(C) Facebook, Inc. and its affiliates.
use crate::worker::SerializedBatchDigestMessage;
use crate::WorkerMessage;
use config::{BatchMetadata, WorkerId};
use crypto::{Digest, PublicKey};
use ed25519_dalek::Digest as _;
use ed25519_dalek::Sha512;
use log::{debug, warn};
use primary::WorkerPrimaryMessage;
use std::convert::TryInto;
use store::Store;
use tokio::sync::mpsc::{Receiver, Sender};

#[cfg(test)]
#[path = "tests/processor_tests.rs"]
pub mod processor_tests;

/// Indicates a serialized `WorkerMessage::Batch` message.
pub type SerializedBatchMessage = Vec<u8>;

/// Hashes and stores batches, it then outputs the batch's digest.
pub struct Processor;

impl Processor {
    pub fn spawn(
        id: WorkerId,
        name: PublicKey,
        mut store: Store,
        mut rx_batch: Receiver<SerializedBatchMessage>,
        tx_digest: Sender<SerializedBatchDigestMessage>,
        own_digest: bool,
    ) {
        tokio::spawn(async move {
            while let Some(batch) = rx_batch.recv().await {
                // Hash the batch.
                let digest = Digest(Sha512::digest(&batch).as_slice()[..32].try_into().unwrap());
                debug!("Processor received batch {:?}", digest);

                // Deserialize the batch to extract metadata
                let batch_metadata = match bincode::deserialize::<WorkerMessage>(&batch) {
                    Ok(WorkerMessage::Batch(batch_struct)) => {
                        // Extract sample transaction IDs, timestamps, and sizes
                        let mut sample_tx_ids = Vec::new();
                        let mut sample_tx_timestamps = Vec::new();
                        let mut sample_tx_sizes = Vec::new();

                        for tx in &batch_struct {
                            if tx.data.len() >= 9 && tx.data[0] == 0u8 {
                                if let Ok(tx_id_bytes) = tx.data[1..9].try_into() {
                                    let tx_id = u64::from_be_bytes(tx_id_bytes);
                                    sample_tx_ids.push(tx_id);
                                    sample_tx_timestamps.push(tx.created_at);
                                    sample_tx_sizes.push(tx.data.len());
                                }
                            }
                        }

                        // Calculate total batch size
                        let batch_size = batch_struct.iter().map(|tx| tx.data.len()).sum();

                        // Calculate average transaction size
                        let avg_transaction_size = if batch_struct.len() > 0 {
                            batch_size / batch_struct.len()
                        } else {
                            0
                        };

                        BatchMetadata {
                            author: name,
                            sample_tx_ids,
                            sample_tx_timestamps,
                            sample_tx_sizes,
                            transaction_count: batch_struct.len(),
                            batch_size,
                            avg_transaction_size,
                        }
                    }
                    _ => {
                        // Fallback: create empty metadata if deserialization fails
                        BatchMetadata {
                            author: name,
                            sample_tx_ids: Vec::new(),
                            sample_tx_timestamps: Vec::new(),
                            sample_tx_sizes: Vec::new(),
                            transaction_count: 0,
                            batch_size: 0,
                            avg_transaction_size: 0,
                        }
                    }
                };

                // Store the batch.
                store.write(digest.to_vec(), batch).await;

                // Deliver the batch's digest and metadata.
                let digest_for_log = digest.clone();
                let message = match own_digest {
                    true => WorkerPrimaryMessage::OurBatch(digest, id, batch_metadata),
                    false => WorkerPrimaryMessage::OthersBatch(digest, id, batch_metadata),
                };
                let message = bincode::serialize(&message)
                    .expect("Failed to serialize our own worker-primary message");
                if tx_digest.try_send(message).is_err() {
                    warn!(
                        "Processor: digest channel full, dropping batch {:?}",
                        digest_for_log
                    );
                }
            }
        });
    }
}
