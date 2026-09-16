from unittest import TestCase

from benchmark.cloudlab_remote import CloudLabBench


class CloudLabEpochStopTests(TestCase):
    def test_finds_largest_metrics_epoch(self):
        latest = CloudLabBench._latest_completed_epoch(
            [
                'epoch_9_slot_304.json',
                'global_state_epoch_100.json',
                'epoch_100_slot_3216.json',
                'epoch_bad_slot_3200.json',
            ]
        )

        self.assertEqual(latest, 100)

    def test_returns_none_without_metrics_file(self):
        latest = CloudLabBench._latest_completed_epoch(
            ['global_state_epoch_100.json', 'notes.txt']
        )

        self.assertIsNone(latest)


if __name__ == '__main__':
    import unittest

    unittest.main()
