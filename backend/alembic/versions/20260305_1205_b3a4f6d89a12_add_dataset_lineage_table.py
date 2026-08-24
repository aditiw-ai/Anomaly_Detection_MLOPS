"""add_dataset_lineage_table

Revision ID: b3a4f6d89a12
Revises: 3c17b76603d0
Create Date: 2026-03-05 12:05:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'b3a4f6d89a12'
down_revision = '3c17b76603d0'
branch_labels = None
depends_on = None


def upgrade():
    # Check if table exists to prevent DuplicateTable error
    bind = op.get_bind()
    insp = sa.inspect(bind)
    if not insp.has_table('dataset_lineage'):
        op.create_table(
            'dataset_lineage',
            sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column('target_dataset_id', postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column('source_dataset_id', postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column('relationship_type', sa.String(length=50), nullable=False),
            sa.Column('lineage_metadata', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            
            sa.ForeignKeyConstraint(['source_dataset_id'], ['datasets.id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['target_dataset_id'], ['datasets.id'], ondelete='CASCADE'),
        )
        
        op.create_index(op.f('ix_dataset_lineage_created_at'), 'dataset_lineage', ['created_at'], unique=False)
        op.create_index(op.f('ix_dataset_lineage_relationship_type'), 'dataset_lineage', ['relationship_type'], unique=False)
        op.create_index(op.f('ix_dataset_lineage_source_dataset_id'), 'dataset_lineage', ['source_dataset_id'], unique=False)
        op.create_index(op.f('ix_dataset_lineage_target_dataset_id'), 'dataset_lineage', ['target_dataset_id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_dataset_lineage_target_dataset_id'), table_name='dataset_lineage')
    op.drop_index(op.f('ix_dataset_lineage_source_dataset_id'), table_name='dataset_lineage')
    op.drop_index(op.f('ix_dataset_lineage_relationship_type'), table_name='dataset_lineage')
    op.drop_index(op.f('ix_dataset_lineage_created_at'), table_name='dataset_lineage')
    op.drop_table('dataset_lineage')
