"""add hierarchical storage support

Revision ID: 20260203_001
Revises: 20260117_001
Create Date: 2026-02-03 13:28:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20260203_001'
down_revision = '001_initial'
branch_labels = None
depends_on = None


def upgrade():
    """Add hierarchical storage support to datasets table and create dataset_lineage table."""
    
    # Add new columns to datasets table
    op.add_column('datasets', sa.Column('dataset_type', sa.String(length=50), nullable=True, server_default='raw'))
    op.add_column('datasets', sa.Column('parent_dataset_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('datasets', sa.Column('split_type', sa.String(length=50), nullable=True))
    op.add_column('datasets', sa.Column('split_job_id', postgresql.UUID(as_uuid=True), nullable=True))
    
    # Create indexes for new columns
    op.create_index(op.f('ix_datasets_dataset_type'), 'datasets', ['dataset_type'], unique=False)
    op.create_index(op.f('ix_datasets_parent_dataset_id'), 'datasets', ['parent_dataset_id'], unique=False)
    op.create_index(op.f('ix_datasets_split_type'), 'datasets', ['split_type'], unique=False)
    op.create_index(op.f('ix_datasets_split_job_id'), 'datasets', ['split_job_id'], unique=False)
    
    # Create dataset_lineage table
    op.create_table(
        'dataset_lineage',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('target_dataset_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('source_dataset_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('relationship_type', sa.String(length=50), nullable=False),
        sa.Column('lineage_metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['source_dataset_id'], ['datasets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['target_dataset_id'], ['datasets.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for dataset_lineage
    op.create_index(op.f('ix_dataset_lineage_created_at'), 'dataset_lineage', ['created_at'], unique=False)
    op.create_index(op.f('ix_dataset_lineage_relationship_type'), 'dataset_lineage', ['relationship_type'], unique=False)
    op.create_index(op.f('ix_dataset_lineage_source_dataset_id'), 'dataset_lineage', ['source_dataset_id'], unique=False)
    op.create_index(op.f('ix_dataset_lineage_target_dataset_id'), 'dataset_lineage', ['target_dataset_id'], unique=False)


def downgrade():
    """Remove hierarchical storage support."""
    
    # Drop dataset_lineage table and its indexes
    op.drop_index(op.f('ix_dataset_lineage_target_dataset_id'), table_name='dataset_lineage')
    op.drop_index(op.f('ix_dataset_lineage_source_dataset_id'), table_name='dataset_lineage')
    op.drop_index(op.f('ix_dataset_lineage_relationship_type'), table_name='dataset_lineage')
    op.drop_index(op.f('ix_dataset_lineage_created_at'), table_name='dataset_lineage')
    op.drop_table('dataset_lineage')
    
    # Drop indexes from datasets table
    op.drop_index(op.f('ix_datasets_split_job_id'), table_name='datasets')
    op.drop_index(op.f('ix_datasets_split_type'), table_name='datasets')
    op.drop_index(op.f('ix_datasets_parent_dataset_id'), table_name='datasets')
    op.drop_index(op.f('ix_datasets_dataset_type'), table_name='datasets')
    
    # Drop columns from datasets table
    op.drop_column('datasets', 'split_job_id')
    op.drop_column('datasets', 'split_type')
    op.drop_column('datasets', 'parent_dataset_id')
    op.drop_column('datasets', 'dataset_type')
