import unittest

from summarize.models.sds.lead import get_lead_summary


class TestLeadSummary(unittest.TestCase):
    def setUp(self):
        self.document = [
            'The first sentence .',
            'Followed by the second .',
            'Finally the third .'
        ]

    def test_max_sentences(self):
        assert self.document[:1] == get_lead_summary(self.document, max_sentences=1)
        assert self.document[:2] == get_lead_summary(self.document, max_sentences=2)
        assert self.document == get_lead_summary(self.document, max_sentences=3)
        assert self.document == get_lead_summary(self.document, max_sentences=4)

    def test_max_token(self):
        assert ['The'] == get_lead_summary(self.document, max_tokens=1)
        assert ['The first sentence .'] == get_lead_summary(self.document, max_tokens=4)
        assert ['The first sentence .', 'Followed'] == get_lead_summary(self.document, max_tokens=5)
        assert ['The first sentence .', 'Followed by the second .', 'Finally the third'] == get_lead_summary(self.document, max_tokens=12)
        assert ['The first sentence .', 'Followed by the second .', 'Finally the third .'] == get_lead_summary(self.document, max_tokens=13)
        assert ['The first sentence .', 'Followed by the second .', 'Finally the third .'] == get_lead_summary(self.document, max_tokens=14)

    def test_max_bytes(self):
        assert ['T'] == get_lead_summary(self.document, max_bytes=1)
        assert ['The first sentence'] == get_lead_summary(self.document, max_bytes=19)
        assert ['The first sentence .'] == get_lead_summary(self.document, max_bytes=20)
        assert ['The first sentence .'] == get_lead_summary(self.document, max_bytes=21)
        assert ['The first sentence .', 'F'] == get_lead_summary(self.document, max_bytes=22)
        assert ['The first sentence .', 'Followed by the second .', 'Finally the third'] == get_lead_summary(self.document, max_bytes=64)
        assert ['The first sentence .', 'Followed by the second .', 'Finally the third .'] == get_lead_summary(self.document, max_bytes=65)
        assert ['The first sentence .', 'Followed by the second .', 'Finally the third .'] == get_lead_summary(self.document, max_bytes=66)

    def test_invalid_arguments(self):
        with self.assertRaises(Exception):
            get_lead_summary(self.document)
        with self.assertRaises(Exception):
            get_lead_summary(self.document, max_sentences=1, max_tokens=1)
        with self.assertRaises(Exception):
            get_lead_summary(self.document, max_sentences=1, max_bytes=1)
        with self.assertRaises(Exception):
            get_lead_summary(self.document, max_tokens=1, max_bytes=1)
        with self.assertRaises(Exception):
            get_lead_summary(self.document, max_sentences=1, max_tokens=1, max_bytes=1)
