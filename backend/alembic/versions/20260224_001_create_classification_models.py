"""Create classification_models table

Revision ID: 20260224_001
Revises: 20260213_001
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260224_001"
down_revision: Union[str, None] = "20260213_001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "classification_models",

        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),

        sa.Column(
            "algorithm_id",
            sa.String(length=100),
            nullable=False,
        ),

        sa.Column(
            "name",
            sa.String(length=255),
            nullable=False,
        ),

        sa.Column(
            "description",
            sa.Text(),
            nullable=True,
        ),

        sa.Column(
            "model_type",
            sa.String(length=50),
            nullable=False,
        ),

        sa.Column(
            "hyperparameters_schema",
            sa.JSON(),
            nullable=False,
        ),

        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),

        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=True,
        ),

        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=True,
        ),

        sa.PrimaryKeyConstraint("id"),

        sa.UniqueConstraint(
            "algorithm_id",
            name="uq_classification_models_algorithm_id",
        ),
    )

    op.create_index(
        "ix_classification_models_algorithm_id",
        "classification_models",
        ["algorithm_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_classification_models_algorithm_id",
        table_name="classification_models",
    )

    op.drop_table("classification_models")