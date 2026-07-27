"""
Monthly report — a downloadable Word/PDF snapshot of a chat's month:
completed and open tasks, decisions logged, and expense-by-category totals.

Deterministic, like digest.build_week(): pulled straight from the task,
decision, and expense stores with no LLM call, so it's instant, free, and
never hallucinates activity that didn't happen. The content dict it builds
matches document_generation.py's schema exactly (see that module's
docstring), so /hisobot reuses the SAME render_docx()/render_pdf() that
/proposal uses — no new rendering code, just a different content source.
"""
from datetime import datetime, timezone

import decisions as decisions_store
import expenses
import tasks

_MONTH_NAMES_UZ = {
    1: "Yanvar", 2: "Fevral", 3: "Mart", 4: "Aprel", 5: "May", 6: "Iyun",
    7: "Iyul", 8: "Avgust", 9: "Sentabr", 10: "Oktabr", 11: "Noyabr", 12: "Dekabr",
}


def _decisions_this_month(entries: list[str], since: datetime) -> list[str]:
    """Each entry starts with 'dd-mm-yyyy — text' (see decisions._entry).
    Entries that don't parse (shouldn't happen, but a hand-edited or legacy
    entry might) are kept rather than silently dropped."""
    out = []
    for e in entries:
        stamp = e.split(" — ", 1)[0].strip()
        try:
            d = datetime.strptime(stamp, "%d-%m-%Y").replace(tzinfo=since.tzinfo)
        except ValueError:
            out.append(e)
            continue
        if d >= since:
            out.append(e)
    return out


async def build_report_content(chat_id: int, user_id: int) -> dict:
    now = tasks.now_local()
    since_local = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    since_utc = since_local.astimezone(timezone.utc)
    month_label = f"{_MONTH_NAMES_UZ.get(now.month, now.month)} {now.year}"

    pending = await tasks.list_tasks(chat_id, {"pending"})
    done = await tasks.list_tasks(chat_id, {"done"})
    done_this_month = tasks.completed_since_list(done, since_utc)
    overdue = tasks.overdue(pending)

    sections = []

    task_lines = [
        f"{'⚠️ ' if t in overdue else ''}{t.title} — "
        f"{datetime.fromisoformat(t.due_at).astimezone(tasks.TZ).strftime('%d-%m')}"
        for t in pending[:25]
    ]
    sections.append({
        "heading": f"📋 Ochiq vazifalar ({len(pending)} ta, shundan kechikkan: {len(overdue)})",
        "body": "\n".join(task_lines) if task_lines else "Ochiq vazifa yo'q.",
    })

    done_lines = [t.title for t in done_this_month[:25]]
    sections.append({
        "heading": f"✅ Bu oy bajarilgan ({len(done_this_month)} ta)",
        "body": "\n".join(done_lines) if done_lines else "Bu oy hech narsa bajarilgan deb belgilanmagan.",
    })

    dec_entries = _decisions_this_month(await decisions_store.get_decisions(chat_id), since_local)
    sections.append({
        "heading": f"📖 Bu oy qabul qilingan qarorlar ({len(dec_entries)} ta)",
        "body": "\n".join(dec_entries[:25]) if dec_entries else "Bu oy qaror qayd etilmagan.",
    })

    table = None
    intro = f"{month_label} oyi bo'yicha avtomatik hisobot."
    if expenses.available():
        since_exp, until_exp, _ = expenses.period_bounds("oy")
        data = await expenses.summary(user_id, since_exp, until_exp)
        if data and data["by_category"]:
            headers = ["Kategoriya", "Miqdor", "Soni"]
            rows = [
                [row["category"], expenses.fmt_amount(row["total"], row["currency"]), str(row["n"])]
                for row in data["by_category"]
            ]
            table = {"title": "💸 Xarajatlar", "headers": headers, "rows": rows}
            total_by_cur: dict[str, object] = {}
            for row in data["by_category"]:
                total_by_cur[row["currency"]] = total_by_cur.get(row["currency"], 0) + row["total"]
            spent_summary = ", ".join(
                expenses.fmt_amount(v, c) for c, v in total_by_cur.items()
            )
            intro += f" Jami xarajat: {spent_summary}."

            budget = await expenses.get_budget(user_id)
            if budget:
                spent_uzs = await expenses.month_to_date_uzs_total(user_id)
                pct = int(spent_uzs / budget * 100)
                intro += f" Byudjet: {expenses.fmt_amount(budget, 'UZS')} ning {pct}% i sarflandi."

    return {
        "title": f"Oylik hisobot — {month_label}",
        "subtitle": None,
        "intro": intro,
        "sections": sections,
        "table": table,
        "closing": "",
    }
