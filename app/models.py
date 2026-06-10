from datetime import datetime
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    prompt_version: Mapped[str] = mapped_column(String(100), default="v1")
    model_name: Mapped[str] = mapped_column(String(100), default="mock-agent")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    runs: Mapped[list["Run"]] = relationship(
        "Run",
        back_populates="experiment",
        cascade="all, delete-orphan"
    )


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    experiment_id: Mapped[int] = mapped_column(Integer, ForeignKey("experiments.id"))

    case_id: Mapped[str] = mapped_column(String(100), index=True)
    task_type: Mapped[str] = mapped_column(String(100))
    query: Mapped[str] = mapped_column(Text)

    expected_tool: Mapped[str | None] = mapped_column(String(100), nullable=True)
    actual_tools: Mapped[str] = mapped_column(Text, default="[]")
    final_answer: Mapped[str] = mapped_column(Text)

    latency_ms: Mapped[int] = mapped_column(Integer)
    tool_accuracy: Mapped[float] = mapped_column(Float)
    keyword_success: Mapped[bool] = mapped_column(Boolean)
    judge_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="runs")
    traces: Mapped[list["Trace"]] = relationship(
        "Trace",
        back_populates="run",
        cascade="all, delete-orphan"
    )


class Trace(Base):
    __tablename__ = "traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("runs.id"))

    step_index: Mapped[int] = mapped_column(Integer)
    node_name: Mapped[str] = mapped_column(String(100))
    input_text: Mapped[str] = mapped_column(Text, default="")
    output_text: Mapped[str] = mapped_column(Text, default="")
    tool_name: Mapped[str | None] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    run: Mapped["Run"] = relationship("Run", back_populates="traces")
