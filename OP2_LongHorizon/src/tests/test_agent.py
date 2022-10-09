from unittest import TestCase
from Agents.Agent import Agent


class TestAgent(TestCase):

    def setUp(self) -> None:
        self.ag = Agent()

    def test_agent_run(self):
        self.ag.run()



