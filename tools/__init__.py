# MRAgent Tools Package
"""
Tool registry — creates and registers all available tools.

Usage:
    from tools import create_tool_registry
    registry = create_tool_registry()
    result = registry.execute("execute_terminal", command="ls -la")
    tools_for_llm = registry.get_openai_tools()
"""

from tools.base import ToolRegistry
from utils.logger import get_logger

logger = get_logger("tools")


def create_tool_registry() -> ToolRegistry:
    """Create a registry with all available tools registered."""
    registry = ToolRegistry()

    # Terminal
    from tools.terminal import TerminalTool
    registry.register(TerminalTool())

    # File manager (multiple tools)
    from tools.file_manager import (
        ReadFileTool, WriteFileTool, ListDirectoryTool,
        MoveFileTool, DeleteFileTool,
    )
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ListDirectoryTool())
    registry.register(MoveFileTool())
    registry.register(DeleteFileTool())

    # Code runner
    from tools.code_runner import CodeRunnerTool
    registry.register(CodeRunnerTool())

    # Screen capture
    from tools.screen import ScreenCaptureTool
    registry.register(ScreenCaptureTool())

    # Browser
    from tools.browser import FetchWebPageTool, SearchWebTool
    registry.register(FetchWebPageTool())
    registry.register(SearchWebTool())

    # PDF Reader
    from tools.pdf_reader import ReadPDFTool
    registry.register(ReadPDFTool())

    # Image generation
    from tools.image_gen import GenerateImageTool
    registry.register(GenerateImageTool())

    # Skills
    from skills.agentmail import AgentMailSkill
    for tool in AgentMailSkill().get_tools():
        registry.register(tool)

    from skills.telegram import TelegramSkill
    for tool in TelegramSkill().get_tools():
        registry.register(tool)

    logger.info(f"Tool registry created with {registry.count} tools")
    return registry
