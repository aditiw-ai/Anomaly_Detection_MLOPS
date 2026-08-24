"""add tuning fields to training_jobs

Revision ID: 20260211_001
Revises: 20260205_001
Create Date: 2026-02-11

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260211_001'
down_revision = '20260205_001'
branch_labels = None
depends_on = None


def upgrade():
    # Add tuning_method and tuning_config columns to training_jobs
    op.add_column('training_jobs', sa.Column('tuning_method', sa.String(length=50), nullable=True))
    op.add_column('training_jobs', sa.Column('tuning_config', sa.JSON(), nullable=True))
    
    # Set default for existing rows
    op.execute("UPDATE training_jobs SET tuning_method = 'manual' WHERE tuning_method IS NULL")
    
    # Make tuning_method non-nullable with default
    op.alter_column('training_jobs', 'tuning_method', nullable=False, server_default='manual')


def downgrade():
    op.drop_column('training_jobs', 'tuning_config')
    op.drop_column('training_jobs', 'tuning_method')
