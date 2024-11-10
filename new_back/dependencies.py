from typing import Annotated, AsyncGenerator, Coroutine
from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession


from db import session_factory
import os


async def session_dependency() -> AsyncGenerator[AsyncSession, None]:
    async with session_factory() as session:
        yield session
        await session.close()
