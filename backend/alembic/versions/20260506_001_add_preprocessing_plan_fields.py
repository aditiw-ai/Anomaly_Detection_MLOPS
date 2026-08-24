"""add preprocessing plan fields

Revision ID: a1f29b17b3e1
Revises: 8b5a6c288c49
Create Date: 2026-05-06 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "a1f29b17b3e1"
down_revision: Union[str, None] = "8b5a6c288c49"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(inspector: sa.Inspector, table: str, name: str) -> bool:
    return any(col["name"] == name for col in inspector.get_columns(table))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _has_column(inspector, "feature_sets", "preprocessing_profile") is False:
        op.add_column("feature_sets", sa.Column("preprocessing_profile", sa.JSON(), nullable=True))
    if _has_column(inspector, "feature_sets", "preprocessing_recommendations") is False:
        op.add_column("feature_sets", sa.Column("preprocessing_recommendations", sa.JSON(), nullable=True))
    if _has_column(inspector, "feature_sets", "preprocessing_decisions") is False:
        op.add_column("feature_sets", sa.Column("preprocessing_decisions", sa.JSON(), nullable=True))
    if _has_column(inspector, "feature_sets", "preprocessing_audit") is False:
        op.add_column("feature_sets", sa.Column("preprocessing_audit", sa.JSON(), nullable=True))
    if _has_column(inspector, "feature_sets", "preprocessing_plan_version") is False:
        op.add_column(
            "feature_sets",
            sa.Column("preprocessing_plan_version", sa.Integer(), nullable=False, server_default="0"),
        )
    if _has_column(inspector, "feature_sets", "preprocessing_committed_at") is False:
        op.add_column("feature_sets", sa.Column("preprocessing_committed_at", sa.DateTime(), nullable=True))
    if _has_column(inspector, "feature_sets", "preprocessing_committed_by") is False:
        op.add_column("feature_sets", sa.Column("preprocessing_committed_by", sa.String(length=255), nullable=True))

    if _has_column(inspector, "training_jobs", "feature_set_id") is False:
        op.add_column("training_jobs", sa.Column("feature_set_id", postgresql.UUID(as_uuid=True), nullable=True))
        op.create_foreign_key(
            "fk_training_jobs_feature_set_id_feature_sets",
            "training_jobs",
            "feature_sets",
            ["feature_set_id"],
            ["id"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    fks = {fk["name"] for fk in inspector.get_foreign_keys("training_jobs")}
    if "fk_training_jobs_feature_set_id_feature_sets" in fks:
        op.drop_constraint("fk_training_jobs_feature_set_id_feature_sets", "training_jobs", type_="foreignkey")
    if _has_column(inspector, "training_jobs", "feature_set_id"):
        op.drop_column("training_jobs", "feature_set_id")

    if _has_column(inspector, "feature_sets", "preprocessing_committed_by"):
        op.drop_column("feature_sets", "preprocessing_committed_by")
    if _has_column(inspector, "feature_sets", "preprocessing_committed_at"):
        op.drop_column("feature_sets", "preprocessing_committed_at")
    if _has_column(inspector, "feature_sets", "preprocessing_plan_version"):
        op.drop_column("feature_sets", "preprocessing_plan_version")
    if _has_column(inspector, "feature_sets", "preprocessing_audit"):
        op.drop_column("feature_sets", "preprocessing_audit")
    if _has_column(inspector, "feature_sets", "preprocessing_decisions"):
        op.drop_column("feature_sets", "preprocessing_decisions")
    if _has_column(inspector, "feature_sets", "preprocessing_recommendations"):
        op.drop_column("feature_sets", "preprocessing_recommendations")
    if _has_column(inspector, "feature_sets", "preprocessing_profile"):
        op.drop_column("feature_sets", "preprocessing_profile")
