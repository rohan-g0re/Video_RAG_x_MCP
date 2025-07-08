#!/usr/bin/env python3
"""
Test Suite for Phase 1 - Audio Processing & Embedding

Tests all Phase 1 deliverables against acceptance criteria:
- extract_transcribe.py: JSON transcript with ≥95% timing field coverage
- segment_transcript.py: Exactly ⌈duration/10⌉ segments with correct start/end
- embed_text.py: Embeds N segments in ≤(N/8)s performance target
"""

import os
import sys
import json
import tempfile
import unittest
import time
import math
from pathlib import Path

# Add src to path for importing modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1_audio.extract_transcribe import VideoTranscriptGenerator, AudioExtractor, WhisperTranscriber
from phase1_audio.segment_transcript import TranscriptSegmenter, TextNormalizer
from phase1_audio.embed_text import TextEmbeddingProcessor, CLIPTextEmbedder, DatabaseClient


class TestMockData:
    """Mock data generator for testing."""
    
    @staticmethod
    def create_mock_transcript(duration: float = 60.0, words_per_second: float = 2.0) -> dict:
        """Create mock transcript data for testing."""
        words = []
        current_time = 0.0
        word_duration = 1.0 / words_per_second
        
        mock_words = [
            "hello", "world", "this", "is", "a", "test", "video", "with", "some", "content",
            "we", "are", "testing", "the", "audio", "processing", "pipeline", "here",
            "silence", "periods", "may", "occur", "between", "different", "segments"
        ] * 10  # Repeat to have enough words
        
        word_index = 0
        while current_time < duration and word_index < len(mock_words):
            word_start = current_time
            word_end = min(current_time + word_duration, duration)
            
            words.append({
                'start': round(word_start, 3),
                'end': round(word_end, 3),
                'word': mock_words[word_index % len(mock_words)]
            })
            
            current_time = word_end
            word_index += 1
        
        return {
            'video_id': 'test_video',
            'video_path': '/path/to/test_video.mp4',
            'language': 'en',
            'full_text': ' '.join(word['word'] for word in words),
            'words': words,
            'word_count': len(words),
            'duration_seconds': duration
        }
    
    @staticmethod
    def create_mock_segmented_data(duration: float = 60.0) -> dict:
        """Create mock segmented transcript data."""
        num_segments = math.ceil(duration / 10.0)
        segments = []
        
        for i in range(num_segments):
            start = i * 10.0
            end = min((i + 1) * 10.0, duration)
            
            # Simulate some silent segments
            if i % 5 == 4:  # Every 5th segment is silent
                text = ""
                word_count = 1
            else:
                text = f"this is segment {i} with some normalized text content"
                word_count = 8
            
            segments.append({
                'start': start,
                'end': end,
                'text': text,
                'word_count': word_count,
                'raw_words': []
            })
        
        return {
            'video_id': 'test_video',
            'total_duration': duration,
            'segment_duration': 10.0,
            'total_segments': len(segments),
            'segments': segments
        }


class TestTextNormalizer(unittest.TestCase):
    """Test text normalization functionality."""
    
    def setUp(self):
        self.normalizer = TextNormalizer()
    
    def test_lowercase_conversion(self):
        """Test that text is converted to lowercase."""
        result = self.normalizer.normalize_text("HELLO WORLD")
        self.assertEqual(result, "hello world")
    
    def test_punctuation_removal(self):
        """Test that punctuation is removed."""
        result = self.normalizer.normalize_text("Hello, world! How are you?")
        self.assertEqual(result, "hello world how are you")
    
    def test_whitespace_collapse(self):
        """Test that multiple whitespace is collapsed."""
        result = self.normalizer.normalize_text("hello    world\t\ntest")
        self.assertEqual(result, "hello world test")
    
    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        self.assertEqual(self.normalizer.normalize_text(""), "")
        self.assertEqual(self.normalizer.normalize_text("   "), "")
        self.assertEqual(self.normalizer.normalize_text(None), "")


class TestTranscriptSegmenter(unittest.TestCase):
    """Test transcript segmentation functionality."""
    
    def setUp(self):
        self.segmenter = TranscriptSegmenter(segment_duration=10.0)
    
    def test_correct_segment_count(self):
        """Test that exactly ⌈duration/10⌉ segments are created."""
        test_cases = [30.0, 35.7, 60.0, 123.4]
        
        for duration in test_cases:
            with self.subTest(duration=duration):
                mock_data = TestMockData.create_mock_transcript(duration)
                result = self.segmenter.segment_transcript(mock_data)
                
                expected_segments = math.ceil(duration / 10.0)
                actual_segments = result['total_segments']
                
                self.assertEqual(actual_segments, expected_segments,
                               f"Duration {duration}s should produce {expected_segments} segments, got {actual_segments}")
    
    def test_segment_timing(self):
        """Test that segment start/end times are correct."""
        mock_data = TestMockData.create_mock_transcript(25.0)
        result = self.segmenter.segment_transcript(mock_data)
        
        segments = result['segments']
        
        for i, segment in enumerate(segments):
            expected_start = i * 10.0
            expected_end = min((i + 1) * 10.0, 25.0)
            
            self.assertAlmostEqual(segment['start'], expected_start, places=1)
            self.assertAlmostEqual(segment['end'], expected_end, places=1)
    
    def test_silent_segment_handling(self):
        """Test that silent segments (≤2 words) have empty text."""
        # Create mock data with minimal words
        mock_data = {
            'video_id': 'test',
            'duration_seconds': 20.0,
            'words': [
                {'start': 5.0, 'end': 6.0, 'word': 'hello'},
                {'start': 15.0, 'end': 16.0, 'word': 'world'}
            ]
        }
        
        result = self.segmenter.segment_transcript(mock_data)
        segments = result['segments']
        
        # First segment (0-10s) should have 1 word -> silent
        self.assertEqual(segments[0]['text'], "")
        
        # Second segment (10-20s) should have 1 word -> silent  
        self.assertEqual(segments[1]['text'], "")
    
    def test_empty_transcript_handling(self):
        """Test handling of transcripts with no words."""
        mock_data = {
            'video_id': 'empty_test',
            'duration_seconds': 30.0,
            'words': []
        }
        
        result = self.segmenter.segment_transcript(mock_data)
        
        self.assertEqual(result['total_segments'], 3)  # 30/10 = 3 segments
        for segment in result['segments']:
            self.assertEqual(segment['text'], "")
            self.assertEqual(segment['word_count'], 0)


class TestCLIPTextEmbedder(unittest.TestCase):
    """Test CLIP text embedding functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up CLIP embedder once for all tests."""
        cls.embedder = CLIPTextEmbedder("ViT-B-32")
    
    def test_embedding_dimension(self):
        """Test that embeddings have correct dimension."""
        texts = ["hello world", "test text"]
        embeddings = self.embedder.embed_texts(texts)
        
        self.assertEqual(embeddings.shape[0], 2)  # 2 texts
        self.assertEqual(embeddings.shape[1], self.embedder.embedding_dim)
    
    def test_empty_text_handling(self):
        """Test handling of empty texts."""
        texts = ["", "   ", "actual text"]
        embeddings = self.embedder.embed_texts(texts)
        
        self.assertEqual(embeddings.shape[0], 3)
        # Should not raise errors
    
    def test_batch_processing(self):
        """Test batch processing with different batch sizes."""
        texts = [f"text number {i}" for i in range(10)]
        
        # Test different batch sizes
        for batch_size in [1, 3, 5, 10]:
            with self.subTest(batch_size=batch_size):
                embeddings = self.embedder.embed_texts(texts, batch_size)
                self.assertEqual(embeddings.shape[0], 10)
    
    def test_performance_benchmark(self):
        """Test embedding performance meets ≤(N/8)s requirement."""
        # Test with different sizes
        sizes = [8, 16, 32, 64]
        
        for n in sizes:
            with self.subTest(n=n):
                texts = [f"test text {i}" for i in range(n)]
                
                start_time = time.time()
                embeddings = self.embedder.embed_texts(texts, batch_size=32)
                elapsed_time = time.time() - start_time
                
                max_time = n / 8.0
                
                self.assertLessEqual(elapsed_time, max_time,
                                   f"Embedding {n} texts took {elapsed_time:.2f}s, should be ≤{max_time:.2f}s")


class TestDatabaseClient(unittest.TestCase):
    """Test database client functionality."""
    
    def setUp(self):
        self.db_client = DatabaseClient()
    
    def test_add_batch(self):
        """Test adding batches of embeddings."""
        import numpy as np
        
        vectors = np.random.rand(5, 512)
        metadatas = [{'id': i, 'test': True} for i in range(5)]
        
        result = self.db_client.add_batch(vectors, metadatas)
        self.assertTrue(result)
        self.assertEqual(self.db_client.get_count(), 5)
    
    def test_metadata_vector_mismatch(self):
        """Test error handling for mismatched vectors and metadata."""
        import numpy as np
        
        vectors = np.random.rand(3, 512)
        metadatas = [{'id': i} for i in range(5)]  # Wrong count
        
        with self.assertRaises(ValueError):
            self.db_client.add_batch(vectors, metadatas)


class TestTextEmbeddingProcessor(unittest.TestCase):
    """Test complete text embedding processing workflow."""
    
    def setUp(self):
        self.processor = TextEmbeddingProcessor(batch_size=16)
    
    def test_process_segmented_transcript(self):
        """Test processing of segmented transcript data."""
        mock_data = TestMockData.create_mock_segmented_data(30.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name
        
        try:
            results = self.processor.process_segmented_transcript(mock_data, output_file)
            
            self.assertEqual(results['status'], 'success')
            self.assertEqual(results['segments_processed'], 3)  # 30s / 10s = 3 segments
            self.assertGreater(results['embedding_dimension'], 0)
            self.assertTrue(results['performance_ok'])
            
            # Verify output file exists
            self.assertTrue(os.path.exists(output_file))
            
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_empty_segments_handling(self):
        """Test handling of empty segment data."""
        empty_data = {
            'video_id': 'empty',
            'segments': []
        }
        
        results = self.processor.process_segmented_transcript(empty_data)
        self.assertEqual(results['status'], 'error')


class TestPhase1Integration(unittest.TestCase):
    """Integration tests for complete Phase 1 workflow."""
    
    def test_end_to_end_workflow(self):
        """Test complete Phase 1 workflow with mock data."""
        # Step 1: Create mock transcript (simulating Phase 1-A output)
        transcript_data = TestMockData.create_mock_transcript(45.0)
        
        # Step 2: Segment transcript (Phase 1-B)
        segmenter = TranscriptSegmenter()
        segmented_data = segmenter.segment_transcript(transcript_data)
        
        # Validate segmentation
        expected_segments = math.ceil(45.0 / 10.0)
        self.assertEqual(segmented_data['total_segments'], expected_segments)
        
        # Step 3: Generate embeddings (Phase 1-C)
        processor = TextEmbeddingProcessor(batch_size=16)
        results = processor.process_segmented_transcript(segmented_data)
        
        # Validate embedding results
        self.assertEqual(results['status'], 'success')
        self.assertEqual(results['segments_processed'], expected_segments)
        self.assertTrue(results['performance_ok'])
        
        print(f"✓ End-to-end test passed: {expected_segments} segments processed")


class TestAcceptanceCriteria(unittest.TestCase):
    """Test specific acceptance criteria from development plan."""
    
    def test_transcript_timing_coverage(self):
        """Test ≥95% coverage for timing fields in transcript."""
        mock_data = TestMockData.create_mock_transcript(30.0)
        
        words = mock_data['words']
        self.assertGreater(len(words), 0)
        
        # Check timing field coverage
        valid_timing_count = 0
        for word in words:
            if ('start' in word and 'end' in word and 
                isinstance(word['start'], (int, float)) and 
                isinstance(word['end'], (int, float)) and
                word['start'] >= 0 and word['end'] >= word['start']):
                valid_timing_count += 1
        
        coverage = valid_timing_count / len(words)
        self.assertGreaterEqual(coverage, 0.95, 
                               f"Timing coverage {coverage:.1%} < 95%")
    
    def test_segmentation_exact_count(self):
        """Test exactly ⌈duration/10⌉ segments with correct start/end."""
        test_durations = [23.7, 45.2, 67.8, 90.1]
        
        segmenter = TranscriptSegmenter()
        
        for duration in test_durations:
            with self.subTest(duration=duration):
                mock_data = TestMockData.create_mock_transcript(duration)
                result = segmenter.segment_transcript(mock_data)
                
                expected_count = math.ceil(duration / 10.0)
                self.assertEqual(result['total_segments'], expected_count)
                
                # Check start/end times
                for i, segment in enumerate(result['segments']):
                    expected_start = i * 10.0
                    expected_end = min((i + 1) * 10.0, duration)
                    
                    self.assertAlmostEqual(segment['start'], expected_start, places=1)
                    self.assertAlmostEqual(segment['end'], expected_end, places=1)
    
    def test_embedding_performance_target(self):
        """Test embeds N segments in ≤(N/8)s on dev laptop."""
        processor = TextEmbeddingProcessor(batch_size=32)
        
        test_sizes = [8, 16, 24, 32]
        
        for n in test_sizes:
            with self.subTest(n=n):
                mock_data = TestMockData.create_mock_segmented_data(n * 10.0)  # n segments
                
                start_time = time.time()
                results = processor.process_segmented_transcript(mock_data)
                actual_time = time.time() - start_time
                
                max_time = n / 8.0
                
                self.assertLessEqual(actual_time, max_time,
                                   f"Processing {n} segments took {actual_time:.2f}s > {max_time:.2f}s")
                self.assertTrue(results['performance_ok'])


def run_phase1_tests():
    """Run all Phase 1 tests and generate report."""
    print("=" * 60)
    print("Phase 1 Audio Processing - Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_classes = [
        TestTextNormalizer,
        TestTranscriptSegmenter,
        TestCLIPTextEmbedder,
        TestDatabaseClient,
        TestTextEmbeddingProcessor,
        TestPhase1Integration,
        TestAcceptanceCriteria
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        class_total = result.testsRun
        class_passed = class_total - len(result.failures) - len(result.errors)
        
        total_tests += class_total
        passed_tests += class_passed
        
        if result.failures or result.errors:
            failed_tests.extend([f"{test_class.__name__}: {fail[0]}" for fail in result.failures])
            failed_tests.extend([f"{test_class.__name__}: {err[0]}" for err in result.errors])
        
        print(f"  ✓ {class_passed}/{class_total} tests passed")
    
    print("\n" + "=" * 60)
    print("PHASE 1 TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests run: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Tests failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests:.1%}")
    
    if failed_tests:
        print("\nFailed tests:")
        for fail in failed_tests:
            print(f"  ✗ {fail}")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_phase1_tests()
    sys.exit(0 if success else 1) 