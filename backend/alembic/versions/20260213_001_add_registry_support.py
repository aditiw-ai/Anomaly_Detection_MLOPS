"""add registry support

Revision ID: 20260213_001
Revises: 20260211_001
Create Date: 2026-02-13 14:45:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '20260213_001'
down_revision = '20260211_001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add columns to feature_sets
    # is_deleted
    op.add_column('feature_sets', sa.Column('is_deleted', sa.Boolean(), server_default='false', nullable=False))
    # deleted_at
    op.add_column('feature_sets', sa.Column('deleted_at', sa.DateTime(), nullable=True))
    # config_hash
    op.add_column('feature_sets', sa.Column('config_hash', sa.String(length=64), nullable=True))
    
    # Add indexes
    op.create_index(op.f('ix_feature_sets_is_deleted'), 'feature_sets', ['is_deleted'], unique=False)
    op.create_index(op.f('ix_feature_sets_config_hash'), 'feature_sets', ['config_hash'], unique=False)


def downgrade() -> None:
    # Remove indexes and columns
    op.drop_index(op.f('ix_feature_sets_config_hash'), table_name='feature_sets')
    op.drop_index(op.f('ix_feature_sets_is_deleted'), table_name='feature_sets')
    op.drop_column('feature_sets', 'config_hash')
    op.drop_column('feature_sets', 'deleted_at')
    op.drop_column('feature_sets', 'is_deleted')
