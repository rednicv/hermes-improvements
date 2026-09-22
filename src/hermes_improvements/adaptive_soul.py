#!/usr/bin/env python3
"""
Adaptive Soul System — Learning and Behavioral Evolution

Provides:
  - AdaptiveSoul: Tracks behavioral adjustments, learns from user feedback
  - StyleLearner: Detects and adapts to user's communication style

Extends the static SOUL.md with dynamic behavioral adaptations learned
from user corrections, praises, and style preferences across sessions.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ─── Data Classes ────────────────────────────────────────────────────

@dataclass
class BehaviorFeedback:
    """User feedback about agent behavior."""
    timestamp: float
    category: str  # "tone", "accuracy", "formatting", "speed", "style"
    sentiment: int  # -1 (bad), 0 (neutral), +1 (good)
    text: str  # User's comment
    context: str  # What the agent did


@dataclass
class AdaptiveRule:
    """A learned behavioral rule."""
    name: str
    category: str  # "tone", "reasoning", "response_format", etc
    priority: float  # 0.0-1.0, higher = apply first
    condition: str  # When this rule applies
    action: str  # What to do
    confidence: float  # How sure we are (0.0-1.0)
    learned_from: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    hits: int = 0  # How often successfully applied


# ─── AdaptiveSoul ────────────────────────────────────────────────────

class AdaptiveSoul:
    """
    Tracks behavioral evolution and applies learned rules.

    Extends static SOUL.md with dynamic behavioral adaptations.
    Persists feedback and rules to disk for cross-session learning.
    """

    def __init__(self, soul_dir: Path):
        self.soul_dir = Path(soul_dir)
        self.soul_dir.mkdir(parents=True, exist_ok=True)

        self.feedback_log_path = self.soul_dir / "soul_feedback.jsonl"
        self.rules_path = self.soul_dir / "soul_adaptive_rules.json"

        self.max_rules = 30  # Maximum number of saved rules
        self.rule_ttl_days = 30  # Rules expire after 30 days
        self.max_rules_in_prompt = 8  # Max 8 rules injected into prompt

        self.feedback_history: List[BehaviorFeedback] = []
        self.adaptive_rules: Dict[str, AdaptiveRule] = {}

        self._load_history()

    # ─── Persistence ──────────────────────────────────────────────

    def _load_history(self):
        """Load feedback history and rules from disk."""
        if self.feedback_log_path.exists():
            try:
                with open(self.feedback_log_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        self.feedback_history.append(BehaviorFeedback(**data))
            except Exception as e:
                logger.warning("Failed to load feedback history: %s", e)

        if self.rules_path.exists():
            try:
                with open(self.rules_path, 'r') as f:
                    rules_data = json.load(f)
                    for rule_name, rule_dict in rules_data.items():
                        self.adaptive_rules[rule_name] = AdaptiveRule(**rule_dict)
            except Exception as e:
                logger.warning("Failed to load adaptive rules: %s", e)

        logger.debug(
            "Loaded %d feedback items, %d rules",
            len(self.feedback_history), len(self.adaptive_rules)
        )

    # ─── Feedback Recording ───────────────────────────────────────

    def record_feedback(
        self,
        category: str,
        sentiment: int,
        text: str,
        context: str = ""
    ):
        """
        Record user feedback about agent behavior.

        Args:
            category: What aspect ("tone", "accuracy", "formatting", etc.)
            sentiment: -1 (negative), 0 (neutral), +1 (positive)
            text: What the user said
            context: What triggered this feedback
        """
        feedback = BehaviorFeedback(
            timestamp=time.time(),
            category=category,
            sentiment=sentiment,
            text=text,
            context=context,
        )

        self.feedback_history.append(feedback)
        self._persist_feedback(feedback)

        # Generate/update adaptive rules
        self._synthesize_rules()

    def _persist_feedback(self, feedback: BehaviorFeedback):
        """Append feedback as JSONL line with file locking."""
        import fcntl
        try:
            with open(self.feedback_log_path, 'a') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(json.dumps(asdict(feedback)) + '\n')
                f.flush()
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError as e:
            logger.error("Failed to persist feedback: %s", e)
        except ImportError:
            # Fallback: no locking if fcntl unavailable (non-Linux)
            try:
                with open(self.feedback_log_path, 'a') as f:
                    f.write(json.dumps(asdict(feedback)) + '\n')
            except OSError as e:
                logger.error("Failed to persist feedback: %s", e)

    # ─── Rule Synthesis ───────────────────────────────────────────

    def _synthesize_rules(self):
        """
        Analyze recent feedback patterns and create/update adaptive rules.
        """
        # Group by category, look at last 20 entries
        by_category: Dict[str, List[BehaviorFeedback]] = {}
        for fb in self.feedback_history[-20:]:
            by_category.setdefault(fb.category, []).append(fb)

        for category, feedbacks in by_category.items():
            if len(feedbacks) < 2:
                continue

            avg_sentiment = sum(f.sentiment for f in feedbacks) / len(feedbacks)

            if avg_sentiment > 0.5:
                # Positive reinforcement — update existing rule or create
                rule_name = f"{category}_positive"
                prev = self.adaptive_rules.get(rule_name)

                # Merge condition with previous
                if prev:
                    condition = prev.condition
                    confidence = min(0.95, prev.confidence + 0.05)
                    hits = prev.hits + 1
                    learned = list(set(prev.learned_from + [f.text[:50] for f in feedbacks]))
                else:
                    condition = self._find_common_pattern([f.context for f in feedbacks if f.sentiment > 0])
                    confidence = min(0.95, 0.3 + len(feedbacks) * 0.1)
                    hits = 1
                    learned = [f.text[:50] for f in feedbacks]

                rule = AdaptiveRule(
                    name=rule_name,
                    category=category,
                    priority=min(0.9, 0.5 + abs(avg_sentiment) * 0.4),
                    condition=condition,
                    action=f"Reinforce {category} behavior from positive feedback",
                    confidence=confidence,
                    learned_from=learned,
                    hits=hits,
                )
                self.adaptive_rules[rule_name] = rule

            elif avg_sentiment < -0.5:
                # Negative correction — update existing rule or create
                rule_name = f"{category}_negative"
                prev = self.adaptive_rules.get(rule_name)

                if prev:
                    condition = prev.condition
                    confidence = min(0.95, prev.confidence + 0.05)
                    hits = prev.hits + 1
                    learned = list(set(prev.learned_from + [f.text[:50] for f in feedbacks]))
                else:
                    condition = self._find_common_pattern([f.context for f in feedbacks if f.sentiment < 0])
                    confidence = min(0.95, 0.3 + len(feedbacks) * 0.1)
                    hits = 1
                    learned = [f.text[:50] for f in feedbacks]

                rule = AdaptiveRule(
                    name=rule_name,
                    category=category,
                    priority=0.95,
                    condition=condition,
                    action=f"Avoid {category} behavior — user feedback",
                    confidence=confidence,
                    learned_from=learned,
                    hits=hits,
                )
                self.adaptive_rules[rule_name] = rule

        self._persist_rules()

    def _find_common_pattern(self, texts: List[str]) -> str:
        """Extract common words from feedback texts."""
        if not texts:
            return "general"

        word_freq: Dict[str, int] = {}
        for text in texts:
            for word in text.split()[:5]:
                word_freq[word] = word_freq.get(word, 0) + 1

        if word_freq:
            common = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return " ".join(w[0] for w in common[:3])

        return "general"

    def _persist_rules(self):
        """Save adaptive rules to JSON, with limits and expiration."""
        try:
            # 1. Remove expired rules (>30 days)
            now = time.time()
            cutoff = now - (self.rule_ttl_days * 86400)
            expired = [
                name for name, rule in self.adaptive_rules.items()
                if rule.created_at < cutoff
            ]
            for name in expired:
                del self.adaptive_rules[name]
            if expired:
                logger.info("AdaptiveSoul: removed %d expired rules (>%d days)", len(expired), self.rule_ttl_days)

            # 2. Keep only top N rules (sorted by priority)
            sorted_rules = sorted(
                self.adaptive_rules.items(),
                key=lambda x: (-x[1].priority, -x[1].confidence)
            )
            self.adaptive_rules = dict(sorted_rules[:self.max_rules])
            if len(sorted_rules) > self.max_rules:
                logger.info("AdaptiveSoul: truncated %d rules to max %d", len(sorted_rules), self.max_rules)

            # 3. Save
            rules_dict = {
                name: asdict(rule) for name, rule in self.adaptive_rules.items()
            }
            tmp = self.rules_path.with_suffix(".tmp")
            with open(tmp, 'w') as f:
                json.dump(rules_dict, f, indent=2)
            tmp.replace(self.rules_path)
        except OSError as e:
            logger.error("Failed to persist rules: %s", e)

    # ─── Rule Access ──────────────────────────────────────────────

    def get_active_rules(self, limit: int = 5) -> List[AdaptiveRule]:
        """Get top active rules sorted by priority and confidence."""
        rules = list(self.adaptive_rules.values())
        rules.sort(key=lambda r: (-r.priority, -r.confidence))
        return rules[:limit]

    def build_soul_extension(self) -> str:
        """
        Build a text fragment to append to SOUL.md for this session.
        Extracts concrete behavioral rules from actual feedback text.
        """
        active = self.get_active_rules()
        if not active and not self.feedback_history:
            return ""

        # Build concrete instructions from real feedback
        concrete = self._extract_concrete_instructions()
        if not concrete:
            return ""

        lines = ["\n## Reguli învățate din feedback (se aplică MANDATORIU)"]
        for instruction in concrete:
            lines.append(f"- {instruction}")

        return "\n".join(lines)

    def _extract_concrete_instructions(self) -> List[str]:
        """
        Extract actionable instructions directly from feedback text history.
        Maps raw feedback phrases to concrete behavioral rules.
        """
        instructions = []
        seen = set()

        # Keyword → concrete rule mapping (multilingual: RO + EN)
        rule_map = [
            (["engleză", "english", "nu engleză", "română", "roman"],
             "Respond ONLY in the user's preferred language. Do not switch languages."),
            (["scurt", "mai scurt", "prea mult text", "verbose", "lung", "concis"],
             "Keep responses SHORT and direct. No unnecessary explanations or padding."),
            (["emoji", "emoticon"],
             "Do not use emojis in responses."),
            (["bold", "**", "formatare", "markdown", "headers", "titluri"],
             "Avoid excessive formatting (bold, headers). Use plain text."),
            (["salut", "bun", "drag", "imi place", "îmi place", "bravo", "bine"],
             "Continue current style — received positive feedback."),
            (["repeta", "repetă", "din nou", "iarăși", "tot timpul"],
             "Do not repeat already known information. Get straight to the point."),
            (["cod", "code", "script", "python", "bash"],
             "Provide complete and functional code, not just fragments."),
        ]

        for fb in self.feedback_history[-30:]:  # last 30 feedback items
            text_lower = fb.text.lower()
            for keywords, instruction in rule_map:
                if any(kw in text_lower for kw in keywords):
                    if instruction not in seen:
                        seen.add(instruction)
                        instructions.append(instruction)

        # Also include any explicit feedback text that looks like a direct rule
        for fb in self.feedback_history[-10:]:
            if fb.sentiment < 0 and len(fb.text) > 10 and fb.text not in seen:
                # Short, direct corrections from user → include verbatim
                if len(fb.text) < 80 and not any(
                    c in fb.text for c in ["test", "Test", "Prima", "A doua", "concurență"]
                ):
                    instructions.append(f"Direct correction: {fb.text}")
                    seen.add(fb.text)

        return instructions[:8]  # max 8 rules in prompt

    def record_behavior(self, outcome: str = ""):
        """Increment hit counters for active rules."""
        for rule in self.get_active_rules():
            rule.hits += 1

    def get_behavioral_stats(self) -> Dict[str, Any]:
        """Get statistics about adaptive behavior."""
        return {
            "total_feedback_items": len(self.feedback_history),
            "active_rules": len(self.adaptive_rules),
            "categories_tracked": list(
                set(f.category for f in self.feedback_history)
            ),
            "average_sentiment": (
                sum(f.sentiment for f in self.feedback_history)
                / len(self.feedback_history)
                if self.feedback_history else 0
            ),
            "top_rules": [
                {"name": r.name, "priority": r.priority, "hits": r.hits}
                for r in sorted(
                    self.adaptive_rules.values(),
                    key=lambda r: r.hits, reverse=True,
                )[:3]
            ],
        }


# ─── StyleLearner ───────────────────────────────────────────────────

class StyleLearner:
    """
    Track user's preferred output style and adapt to it.

    Detects:
      - Verbosity preference (concise/medium/verbose)
      - Language (en/ro/etc.)
      - Emoji usage preference
      - Formality level
      - List/bullet preference
    """

    def __init__(self):
        self.observed_styles = {
            "verbosity": "medium",
            "language": "en",
            "use_examples": False,
            "prefer_lists": False,
            "emoji_usage": False,
            "formality": "neutral",
        }
        self._message_count = 0

    def analyze_user_message(self, message: str):
        """
        Infer user's style preferences from their message.

        Args:
            message: The user's text message
        """
        self._message_count += 1

        if not message:
            return

        # Detect Romanian
        ro_chars = set("ăâîșț")
        if any(c in message.lower() for c in ro_chars):
            self.observed_styles["language"] = "ro"

        # Detect emoji usage
        emoji_ranges = [
            (0x1F300, 0x1F9FF),  # Misc symbols, emoticons
            (0x2600, 0x26FF),    # Misc symbols
            (0x2700, 0x27BF),    # Dingbats
        ]
        for ch in message:
            cp = ord(ch)
            for lo, hi in emoji_ranges:
                if lo <= cp <= hi:
                    self.observed_styles["emoji_usage"] = True
                    break

        # Detect verbosity from message length
        words = message.split()
        if len(words) < 15:
            self.observed_styles["verbosity"] = "concise"
        elif len(words) > 80:
            self.observed_styles["verbosity"] = "verbose"
        else:
            self.observed_styles["verbosity"] = "medium"

        # Detect list preference
        if any(line.strip().startswith(('-', '*', '+', '1.', '#'))
               for line in message.split('\n')):
            self.observed_styles["prefer_lists"] = True

    def get_style_guidance(self) -> str:
        """Build style guidance text for inclusion in system prompt."""
        parts = []

        verb = self.observed_styles["verbosity"]
        if verb == "concise":
            parts.append("Keep responses SHORT and direct. Avoid fluff.")
        elif verb == "verbose":
            parts.append("Provide detailed explanations. Expand on concepts.")
        else:
            parts.append("Balance brevity with completeness.")

        if self.observed_styles["emoji_usage"]:
            parts.append("Use emojis where helpful.")
        else:
            parts.append("Avoid emojis.")

        if self.observed_styles["language"] == "ro":
            parts.append("Respond in Romanian.")

        if self.observed_styles["prefer_lists"]:
            parts.append("Use bullet lists and structured formatting.")

        return " ".join(parts) if parts else ""
