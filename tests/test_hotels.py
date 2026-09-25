import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import hotels


def answers(label="positive"):
    return {topic: {"type": "choice", "choice": label, "confidence": .95,
                   "probabilities": {key: float(key == label) for key in question["criteria"]}}
            for topic, question in hotels.QUESTIONS.items()}


class DemoTests(unittest.TestCase):
    def test_saved_mode_skips_apify_and_preserves_full_review_pool(self):
        rows = [{"hotel": next(iter(hotels.HOTELS)), "id": str(i), "title": "", "text": "Quiet"} for i in range(2)]
        def respond(url, key, body):
            self.assertIn("openrouter.ai", url)
            if "decisions" in url:
                return {"answers": answers(), "usage": {"cost": .001}}
            return {"choices": [{"message": {"content": '{"reason":"Best-supported pick."}'}}], "usage": {"cost": .001}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(hotels, "HERE", Path(tmp)), \
             patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}, clear=True), \
             patch("hotels.collect") as collect, patch("hotels.request", side_effect=respond):
            output = Path(tmp)/"data/results.json"
            output.parent.mkdir()
            output.write_text(json.dumps({"apify_run": {"usageTotalUsd": 3}, "results": [{"review": r} for r in rows]}))
            for argv, expected in [(["--saved", "--reviews", "1"], 1), (["--saved"], 2)]:
                with patch("sys.argv", ["hotels.py", *argv]), redirect_stdout(io.StringIO()) as console:
                    hotels.main()
                saved = json.loads(output.read_text())
                self.assertEqual(len(saved["results"]), expected)
                self.assertEqual(len(saved["reviews"]), 2)
                self.assertTrue(saved["reused_reviews"])
                self.assertIn("Apify: $0 (saved reviews)", console.getvalue())
            collect.assert_not_called()

    def test_selection_skips_blank_and_duplicate_reviews_and_balances_hotels(self):
        rows = [{"hotelId": name, "id": str(i), "customData": {"hotelName": name},
                 "likedText": "Quiet", "startUrl": "https://example.com"}
                for name in hotels.HOTELS for i in range(3)]
        rows += [rows[0], dict(rows[0], id="empty", likedText=None)]
        selected = hotels.select_reviews(rows, 4)
        self.assertEqual([r["hotel"] for r in selected[:3]], list(hotels.HOTELS))
        self.assertEqual(len(selected), 4)
        self.assertEqual(len(hotels.select_reviews(rows, 100)), 9)

    def test_uncertainty_and_invalid_responses(self):
        self.assertFalse(hotels.needs_review(answers()))
        self.assertTrue(hotels.needs_review(answers("unclear")))
        uncertain = answers()
        uncertain["wifi"]["confidence"] = .5
        self.assertTrue(hotels.needs_review(uncertain))
        uncertain["noise"]["probabilities"]["positive"] = float("nan")
        with self.assertRaises(ValueError):
            hotels.needs_review(uncertain)

    def test_main_collects_classifies_and_presents(self):
        row = {"hotel": next(iter(hotels.HOTELS)), "id": "1", "title": "", "text": "Quiet room"}
        response = {"answers": answers(), "usage": {"cost": .001}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(hotels, "HERE", Path(tmp)), \
             patch("sys.argv", ["hotels.py", "--reviews", "1"]), patch("hotels.load_keys"), \
             patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}), \
             patch("hotels.collect", return_value=([row], {"usageTotalUsd": .01})), \
             patch("hotels.request", side_effect=[response, {"choices": [{"message": {"content": '{"reason": "Best-supported pick in this sample."}'}}], "usage": {"cost": .002}}]), redirect_stdout(io.StringIO()) as output:
            hotels.main()
            saved = json.loads((Path(tmp)/"data/results.json").read_text())
            self.assertEqual(len(saved["results"]), 1)
            self.assertIn("Wi-Fi bad/good", output.getvalue())
            self.assertIn("$0.001000", output.getvalue())
            self.assertEqual(saved["verdict"]["winner"], row["hotel"])
            self.assertIn("Best hotel:", output.getvalue())

    def test_gpt_recovers_jev_failure_and_retains_both_responses(self):
        original = {"review": {"hotel": next(iter(hotels.HOTELS)), "title": "", "text": "Quiet but Wi-Fi broke"},
                    "needs_review": True, "error": "OSError"}
        response = {"choices": [{"message": {"content": '{"wifi":"complaint","noise":"positive"}'}}]}
        with patch("hotels.gpt", return_value=response):
            result = hotels.review_with_gpt(original)
        self.assertFalse(result["needs_review"])
        self.assertEqual(result["labels"]["wifi"], "complaint")
        self.assertIn("error", result)  # Original Jev failure remains auditable.
        self.assertIn("llm_response", result)
        summary = hotels.summarize([result])[original["review"]["hotel"]]
        self.assertEqual(summary["counts"]["gpt_reviewed"], 1)
        self.assertEqual(summary["counts"]["unresolved"], 0)

    def test_failed_or_ambiguous_gpt_is_not_silently_resolved(self):
        original = {"review": {"title": "", "text": "Maybe noisy"}, "needs_review": True}
        with patch("hotels.gpt", side_effect=OSError("offline")):
            self.assertIn("llm_error", hotels.review_with_gpt(original))
        response = {"choices": [{"message": {"content": '{"wifi":"unmentioned","noise":"unclear"}'}}]}
        with patch("hotels.gpt", return_value=response):
            self.assertTrue(hotels.review_with_gpt(original)["needs_review"])

    def test_equal_weights_and_missing_evidence(self):
        a, b, c = hotels.HOTELS
        def row(hotel, wifi, noise):
            return {"review": {"hotel": hotel}, "needs_review": False, "labels": {"wifi": wifi, "noise": noise}}
        results = [row(a, "positive", "complaint"), row(b, "complaint", "positive"), row(c, "positive", "unmentioned")]
        summary = hotels.summarize(results)
        self.assertEqual(summary[a]["score"], summary[b]["score"])
        self.assertEqual(summary[a]["score"], 50)
        self.assertIsNone(summary[c]["score"])
        results.append(row(a, "unmentioned", "unmentioned"))
        self.assertEqual(hotels.summarize(results)[a]["score"], summary[a]["score"])

    def test_no_evidence_does_not_invent_a_winner(self):
        with patch("hotels.gpt") as model:
            result, response = hotels.verdict(hotels.summarize([]))
        self.assertIsNone(result["winner"])
        self.assertIsNone(response)
        model.assert_not_called()

    def test_failed_call_is_flagged_for_review(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}), patch("hotels.request", side_effect=OSError("offline")):
            result = hotels.classify({"title": "", "text": "Quiet"})
        self.assertTrue(result["needs_review"])
        self.assertIn("error", result)

    def test_main_sends_only_uncertain_rows_to_gpt(self):
        rows = [{"hotel": next(iter(hotels.HOTELS)), "id": str(i), "title": "", "text": text}
                for i, text in enumerate(("Quiet room", "Ambiguous review"))]
        def respond(url, key, body):
            if "decisions" in url:
                labels = answers()
                if body["state"]["review"]["text"] == "Ambiguous review":
                    labels["noise"]["confidence"] = .5
                return {"answers": labels, "usage": {"cost": .001}}
            data = json.loads(body["messages"][1]["content"])
            if "review" in data:
                self.assertIn("Jev classifications", console.getvalue())
                self.assertNotIn("After GPT Mini review", console.getvalue())
                self.assertEqual(data["review"]["text"], "Ambiguous review")
                content = {"wifi": "unmentioned", "noise": "complaint"}
            else:
                self.assertIn("After GPT Mini review", console.getvalue())
                self.assertNotIn("Final scores", console.getvalue())
                content = {"reason": "The best-supported hotel in this sample."}
            return {"choices": [{"message": {"content": json.dumps(content)}}], "usage": {"cost": .002}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(hotels, "HERE", Path(tmp)), \
             patch("sys.argv", ["hotels.py", "--reviews", "2"]), patch("hotels.load_keys"), \
             patch.dict("os.environ", {"OPENROUTER_API_KEY": "test"}), \
             patch("hotels.collect", return_value=(rows, {})), \
             patch("hotels.request", side_effect=respond) as api, redirect_stdout(io.StringIO()) as console:
            hotels.main()
            saved = json.loads((Path(tmp)/"data/results.json").read_text())
            self.assertEqual(api.call_count, 4)  # 2 Jev, 1 fallback, 1 verdict.
            self.assertNotIn("llm_response", saved["results"][0])
            counts = saved["summary"][rows[0]["hotel"]]["counts"]
            self.assertEqual(counts["noise_complaint"], 1)
            self.assertEqual(counts["gpt_reviewed"], 1)
            self.assertEqual(counts["unresolved"], 0)
            text = console.getvalue()
            stages = ["Jev classifications", "Reviewing 1 uncertain", "After GPT Mini review", "Preparing the final verdict", "Final scores", "Best hotel:"]
            self.assertEqual(sorted(text.index(stage) for stage in stages), [text.index(stage) for stage in stages])


if __name__ == "__main__":
    unittest.main()
