from collections import defaultdict

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    print("NLTK not available. Install with: pip install nltk")
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("Rouge-score not available. Install with: pip install rouge-score")
    ROUGE_AVAILABLE = False


class CaptionMetrics:
    def __init__(self):
        self.smoothing = SmoothingFunction().method1 if NLTK_AVAILABLE else None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) if ROUGE_AVAILABLE else None

    def calculate_bleu(self, reference, candidate):
        if not NLTK_AVAILABLE or not reference.strip() or not candidate.strip():
            return 0.0
        try:
            return sentence_bleu([reference.split()], candidate.split(), smoothing_function=self.smoothing)
        except Exception:
            return 0.0

    def calculate_meteor(self, reference, candidate):
        if not NLTK_AVAILABLE or not reference.strip() or not candidate.strip():
            return 0.0
        try:
            return meteor_score([reference.split()], candidate.split())
        except Exception:
            return 0.0

    def calculate_all_metrics(self, reference, candidate):
        return {
            'bleu': self.calculate_bleu(reference, candidate),
            'meteor': self.calculate_meteor(reference, candidate),
        }


