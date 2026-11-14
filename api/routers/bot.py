"""
Paper bot endpoints: start, stop, kill switch, and status. Paper trading only.

Off unless BOT_ENABLED=true. With BOT_ADMIN_TOKEN set, every endpoint here except
/bot/info needs the token in an X-Bot-Token header.
"""

import hmac
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import BaseModel

from api.config import settings
from api.services.bot import runner as bot
from api.services.bot.streams import StreamNotAvailable
from api.services.paper_broker import PAPER_URL, BrokerError, BrokerNotConfigured, LiveTradingRefused, assert_paper_url

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/bot", tags=["Paper bot"])

DISABLED = "The paper bot is off. Set BOT_ENABLED=true in .env and restart the API."
BAD_TOKEN = "This needs the bot token. Enter the BOT_ADMIN_TOKEN value on the Bot page."


def require_bot_access(x_bot_token: Optional[str] = Header(None)) -> None:
    if not settings.BOT_ENABLED:
        raise HTTPException(status_code=403, detail=DISABLED)
    if settings.BOT_ADMIN_TOKEN and not hmac.compare_digest(x_bot_token or "", settings.BOT_ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail=BAD_TOKEN)


class KillRequest(BaseModel):
    close_positions: bool = False


def _paper_url_ok() -> bool:
    try:
        assert_paper_url(settings.ALPACA_TRADING_URL)
        return True
    except LiveTradingRefused:
        return False


@router.get("/info")
def get_info():
    """What the page needs before it can call anything else. No secrets."""
    return {
        "enabled": settings.BOT_ENABLED,
        "token_required": bool(settings.BOT_ADMIN_TOKEN),
        "paper_url": PAPER_URL,
        "paper_url_ok": _paper_url_ok(),
        "alpaca_configured": bool(settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY),
        "polygon_configured": bool(settings.POLYGON_API_KEY),
        "polygon_realtime": settings.POLYGON_REALTIME,
    }


@router.get("/strategies", dependencies=[Depends(require_bot_access)])
def get_strategies():
    return {"strategies": bot.strategy_catalog()}


@router.get("/status", dependencies=[Depends(require_bot_access)])
def get_status(response: Response, runner: bot.BotRunner = Depends(bot.get_runner)):
    response.headers["Cache-Control"] = "no-store"
    return runner.status()


@router.post("/start", dependencies=[Depends(require_bot_access)])
def post_start(config: bot.BotConfig, runner: bot.BotRunner = Depends(bot.get_runner)):
    try:
        runner.start(config)
    except bot.AlreadyRunning as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except (LiveTradingRefused, BrokerNotConfigured, StreamNotAvailable) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"running": True, "strategy": config.strategy}


@router.post("/stop", dependencies=[Depends(require_bot_access)])
def post_stop(runner: bot.BotRunner = Depends(bot.get_runner)):
    was_running = runner.stop()
    return {"running": False, "was_running": was_running}


@router.post("/kill", dependencies=[Depends(require_bot_access)])
def post_kill(body: KillRequest = KillRequest(), runner: bot.BotRunner = Depends(bot.get_runner)):
    try:
        result = runner.kill(close_positions=body.close_positions)
    except (LiveTradingRefused, BrokerNotConfigured) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except BrokerError as exc:
        # The bot is already stopped and blocked at this point; only the cancel call failed.
        logger.warning("Kill switch cancel failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"The bot stopped, but canceling orders failed: {exc}")
    return {"running": False, **result}
