"""Message bus module for decoupled channel-agent communication."""

from mragent.bus.events import InboundMessage, OutboundMessage
from mragent.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
