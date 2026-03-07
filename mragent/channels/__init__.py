"""Chat channels module with plugin architecture."""

from mragent.channels.base import BaseChannel
from mragent.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]
