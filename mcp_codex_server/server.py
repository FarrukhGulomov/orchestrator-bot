#!/usr/bin/env python3
import asyncio, os
from openai import AsyncOpenAI
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

app = Server("codex-tool")
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="codex_complete",
            description=(
                "OpenAI Codex / GPT-4o orqali kod yozadi yoki to'ldiradi. "
                "Claude hal qila olmagan yoki ikkinchi fikr kerak bo'lgan "
                "kod bo'laklarini shu tool orqali yuboradi."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Kod tavsifi yoki mavjud kod + instruksiya"},
                    "language": {"type": "string", "default": "python"},
                    "model": {"type": "string", "default": "gpt-4o"},
                },
                "required": ["prompt"],
            },
        ),
        types.Tool(
            name="codex_review",
            description="Berilgan kodni OpenAI orqali review qiladi, muammolarni topadi.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "focus": {"type": "string", "description": "security | performance | style | bugs", "default": "bugs"},
                },
                "required": ["code"],
            },
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "codex_complete":
        response = await client.chat.completions.create(
            model=arguments.get("model", "gpt-4o"),
            messages=[
                {"role": "system", "content": f"You are an expert {arguments.get('language','python')} developer. Return only clean, production-ready code."},
                {"role": "user", "content": arguments["prompt"]},
            ],
            temperature=0.2,
        )
        result = response.choices[0].message.content

    elif name == "codex_review":
        focus = arguments.get("focus", "bugs")
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Senior code reviewer. Focus: {focus}. Be concise, list issues with line references."},
                {"role": "user", "content": f"Review:\n\n