"""Regression tests for the ShopSimulator item-page option state."""

from __future__ import annotations

import unittest
from collections import defaultdict
from unittest.mock import patch

from web_agent_site.envs import web_agent_text_env


class ItemPageOptionRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.asin = "123456789012"
        self.server = web_agent_text_env.SimServer.__new__(
            web_agent_text_env.SimServer
        )
        self.server.base_url = "http://127.0.0.1:5700"
        self.server.show_attrs = False
        self.server.product_item_dict = {
            self.asin: {
                "Price": 1999.0,
                "customization_options": {
                    "颜色分类": [
                        {"value": "白色", "price": 1999.0},
                        {"value": "黑色", "price": 2099.0},
                    ]
                },
            }
        }
        self.server.user_sessions = {
            "session": {
                "actions": defaultdict(int),
                "asins": set(),
                "options": {},
                "keywords": ["query"],
                "page": 1,
                "goal": {"instruction_text": "test instruction"},
            }
        }

    @patch.object(
        web_agent_text_env,
        "evaluate_candidate_eligibility",
        return_value={"known_valid": False},
    )
    @patch.object(web_agent_text_env, "map_action_to_html", return_value="<html />")
    def test_open_product_without_an_option_is_rendered(
        self,
        render,
        _evaluate_candidate,
    ) -> None:
        html, _url = self.server.item_page(
            "session",
            clickable_name=self.asin,
            text_to_clickable={self.asin: {"class": ["product-link"]}},
        )
        self.assertEqual(html, "<html />")
        self.assertIsNone(render.call_args.kwargs["selected_option"])
        self.assertIsNone(render.call_args.kwargs["selected_price"])
        self.assertEqual(
            self.server.user_sessions["session"]["price_resolution"]["method"],
            "effective_price_axis_unselected",
        )

    @patch.object(
        web_agent_text_env,
        "evaluate_candidate_eligibility",
        return_value={"known_valid": False},
    )
    @patch.object(web_agent_text_env, "map_action_to_html", return_value="<html />")
    def test_selected_option_and_price_are_rendered(
        self,
        render,
        _evaluate_candidate,
    ) -> None:
        self.server.item_page(
            "session",
            clickable_name=self.asin,
            text_to_clickable={self.asin: {"class": ["product-link"]}},
        )
        self.server.item_page(
            "session",
            clickable_name="白色",
            text_to_clickable={"白色": {"name": "颜色分类"}},
        )
        self.assertEqual(render.call_args.kwargs["selected_option"], "白色")
        self.assertEqual(render.call_args.kwargs["selected_price"], 1999.0)
        self.assertEqual(
            render.call_args.kwargs["options"],
            {"颜色分类": "白色"},
        )


if __name__ == "__main__":
    unittest.main()
