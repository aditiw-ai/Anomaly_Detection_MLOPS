"""Add hyperparameter_presets table

Revision ID: 20260205_001
Revises: 20260203_001_add_hierarchical_storage_support
Create Date: 2026-02-05 13:20:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260205_001'
down_revision = '20260203_001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create hyperparameter_presets table
    op.create_table(
        'hyperparameter_presets',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('algorithm_id', sa.String(length=100), nullable=False),
        sa.Column('preset_name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('hyperparameters', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index(
        'ix_hyperparameter_presets_algorithm_id',
        'hyperparameter_presets',
        ['algorithm_id'],
        unique=False
    )
    
    op.create_index(
        'ix_preset_algorithm_name',
        'hyperparameter_presets',
        ['algorithm_id', 'preset_name'],
        unique=False
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_preset_algorithm_name', table_name='hyperparameter_presets')
    op.drop_index('ix_hyperparameter_presets_algorithm_id', table_name='hyperparameter_presets')
    
    # Drop table
    op.drop_table('hyperparameter_presets')
