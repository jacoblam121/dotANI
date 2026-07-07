use crossbeam_channel::{Receiver, Sender, bounded};
use needletail::{Sequence, parse_fastx_file};
use std::path::Path;

pub struct ReaderGate {
    tokens: Sender<()>,
    releases: Receiver<()>,
}

pub struct ReaderPermit {
    token: Option<()>,
    releases: Sender<()>,
}

impl ReaderGate {
    pub fn new(limit: usize) -> Self {
        assert!(limit > 0, "reader gate limit must be greater than zero");

        let (tokens, releases) = bounded(limit);
        for _ in 0..limit {
            tokens.send(()).expect("reader gate token channel closed");
        }

        Self { tokens, releases }
    }

    pub fn acquire(&self) -> ReaderPermit {
        let token = self
            .releases
            .recv()
            .expect("reader gate token channel closed");
        ReaderPermit {
            token: Some(token),
            releases: self.tokens.clone(),
        }
    }
}

impl Drop for ReaderPermit {
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            let _ = self.releases.send(token);
        }
    }
}

// Read merged sequences from a genome file into single u8 vector
pub fn read_merge_seq(file_name: &Path) -> Vec<u8> {
    let mut fna_seqs = Vec::<u8>::new();

    let mut fastx_reader = parse_fastx_file(file_name).expect("Opening .fna files failed");
    while let Some(record) = fastx_reader.next() {
        let seqrec = record.expect("invalid record");
        let norm_seq = seqrec.normalize(false);

        fna_seqs.push(b'N');
        fna_seqs.extend_from_slice(norm_seq.as_ref());
    }

    fna_seqs
}

#[cfg(test)]
mod tests {
    use super::ReaderGate;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::Duration;

    #[test]
    fn reader_gate_limits_concurrent_readers() {
        let limit = 2;
        let gate = Arc::new(ReaderGate::new(limit));
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));

        std::thread::scope(|scope| {
            for _ in 0..8 {
                let gate = Arc::clone(&gate);
                let active = Arc::clone(&active);
                let max_active = Arc::clone(&max_active);

                scope.spawn(move || {
                    let _permit = gate.acquire();
                    let now_active = active.fetch_add(1, Ordering::SeqCst) + 1;
                    max_active.fetch_max(now_active, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(10));
                    active.fetch_sub(1, Ordering::SeqCst);
                });
            }
        });

        assert!(max_active.load(Ordering::SeqCst) <= limit);
    }

    #[test]
    fn reader_gate_permit_drop_releases_next_acquire() {
        let gate = Arc::new(ReaderGate::new(1));
        let first = gate.acquire();

        let acquired = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|scope| {
            let gate = Arc::clone(&gate);
            let acquired_in_thread = Arc::clone(&acquired);
            scope.spawn(move || {
                let _permit = gate.acquire();
                acquired_in_thread.store(1, Ordering::SeqCst);
            });

            std::thread::sleep(Duration::from_millis(10));
            assert_eq!(acquired.load(Ordering::SeqCst), 0);
            drop(first);
        });

        assert_eq!(acquired.load(Ordering::SeqCst), 1);
    }
}
