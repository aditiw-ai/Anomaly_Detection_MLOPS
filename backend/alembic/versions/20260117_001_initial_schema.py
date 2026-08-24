"""Initial schema with datasets, features, models

Revision ID: 001_initial
Revises: 
Create Date: 2026-01-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create datasets table
    op.create_table(
        'datasets',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('storage_path', sa.String(500), nullable=False),
        sa.Column('file_format', sa.String(50), nullable=True),
        sa.Column('file_size_bytes', sa.BigInteger(), nullable=True),
        sa.Column('row_count', sa.Integer(), nullable=True),
        sa.Column('column_count', sa.Integer(), nullable=True),
        sa.Column('schema', postgresql.JSON(), nullable=True),
        sa.Column('statistics', postgresql.JSON(), nullable=True),
        sa.Column('status', sa.String(50), nullable=True),
        sa.Column('parent_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_datasets_name', 'datasets', ['name'])
    op.create_index('ix_datasets_status', 'datasets', ['status'])
    op.create_index('ix_datasets_created_at', 'datasets', ['created_at'])
    
    # Create feature_sets table
    op.create_table(
        'feature_sets',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('dataset_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('config', postgresql.JSON(), nullable=False),
        sa.Column('all_features', postgresql.JSON(), nullable=True),
        sa.Column('selected_features', postgresql.JSON(), nullable=True),
        sa.Column('selection_report', postgresql.JSON(), nullable=True),
        sa.Column('storage_path', sa.String(500), nullable=True),
        sa.Column('input_rows', sa.Integer(), nullable=True),
        sa.Column('feature_count', sa.Integer(), nullable=True),
        sa.Column('selected_feature_count', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(50), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('processing_time_seconds', sa.Integer(), nullable=True),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_feature_sets_status', 'feature_sets', ['status'])
    
    # Create ml_models table
    op.create_table(
        'ml_models',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('algorithm', sa.String(100), nullable=False),
        sa.Column('hyperparameters', postgresql.JSON(), nullable=False),
        sa.Column('feature_set_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('feature_set_version', sa.String(50), nullable=True),
        sa.Column('storage_path', sa.String(500), nullable=False),
        sa.Column('onnx_path', sa.String(500), nullable=True),
        sa.Column('model_size_bytes', sa.BigInteger(), nullable=True),
        sa.Column('checksum', sa.String(64), nullable=True),
        sa.Column('metrics', postgresql.JSON(), nullable=False),
        sa.Column('feature_names', postgresql.JSON(), nullable=True),
        sa.Column('feature_importance', postgresql.JSON(), nullable=True),
        sa.Column('status', sa.String(50), nullable=True),
        sa.Column('promoted_at', sa.DateTime(), nullable=True),
        sa.Column('archived_at', sa.DateTime(), nullable=True),
        sa.Column('archived_reason', sa.Text(), nullable=True),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['feature_set_id'], ['feature_sets.id']),
    )
    op.create_index('ix_ml_models_status', 'ml_models', ['status'])
    op.create_index('ix_ml_models_created_at', 'ml_models', ['created_at'])
    
    # Create baselines table
    op.create_table(
        'baselines',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('threshold', sa.Float(), nullable=False),
        sa.Column('operator', sa.String(10), nullable=False),
        sa.Column('is_active', sa.String(10), nullable=True),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['model_id'], ['ml_models.id'], ondelete='CASCADE'),
    )
    
    # Create training_jobs table
    op.create_table(
        'training_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('feature_set_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('algorithm', sa.String(100), nullable=False),
        sa.Column('hyperparameters', postgresql.JSON(), nullable=False),
        sa.Column('status', sa.String(50), nullable=True),
        sa.Column('progress', sa.Float(), nullable=True),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['feature_set_id'], ['feature_sets.id']),
        sa.ForeignKeyConstraint(['model_id'], ['ml_models.id']),
    )
    op.create_index('ix_training_jobs_status', 'training_jobs', ['status'])
    
    # Create predictions table
    op.create_table(
        'predictions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('transaction_id', sa.String(100), nullable=True),
        sa.Column('input_features', postgresql.JSON(), nullable=False),
        sa.Column('prediction', sa.Integer(), nullable=False),
        sa.Column('fraud_score', sa.Float(), nullable=False),
        sa.Column('explanation', postgresql.JSON(), nullable=True),
        sa.Column('response_time_ms', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    op.create_index('ix_predictions_model_created', 'predictions', ['model_id', 'created_at'])
    op.create_index('ix_predictions_created_at', 'predictions', ['created_at'])
    
    # Create drift_metrics table
    op.create_table(
        'drift_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('drift_type', sa.String(50), nullable=False),
        sa.Column('feature_name', sa.String(100), nullable=True),
        sa.Column('metric_name', sa.String(50), nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('threshold', sa.Float(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('computed_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['model_id'], ['ml_models.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_drift_metrics_model_computed', 'drift_metrics', ['model_id', 'computed_at'])
    
    # Create bias_metrics table
    op.create_table(
        'bias_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('protected_attribute', sa.String(100), nullable=False),
        sa.Column('metric_name', sa.String(50), nullable=False),
        sa.Column('group_values', postgresql.JSON(), nullable=False),
        sa.Column('overall_value', sa.Float(), nullable=True),
        sa.Column('threshold', sa.Float(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('computed_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['model_id'], ['ml_models.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_bias_metrics_computed', 'bias_metrics', ['computed_at'])
    
    # Create alerts table
    op.create_table(
        'alerts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('model_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('alert_type', sa.String(50), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('details', postgresql.JSON(), nullable=True),
        sa.Column('status', sa.String(20), nullable=True),
        sa.Column('acknowledged_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('acknowledged_at', sa.DateTime(), nullable=True),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['model_id'], ['ml_models.id'], ondelete='SET NULL'),
    )
    op.create_index('ix_alerts_status_created', 'alerts', ['status', 'created_at'])
    op.create_index('ix_alerts_created_at', 'alerts', ['created_at'])


def downgrade() -> None:
    op.drop_index('ix_alerts_created_at')
    op.drop_index('ix_alerts_status_created')
    op.drop_table('alerts')
    op.drop_index('ix_bias_metrics_computed')
    op.drop_table('bias_metrics')
    op.drop_index('ix_drift_metrics_model_computed')
    op.drop_table('drift_metrics')
    op.drop_index('ix_predictions_created_at')
    op.drop_index('ix_predictions_model_created')
    op.drop_table('predictions')
    op.drop_index('ix_training_jobs_status')
    op.drop_table('training_jobs')
    op.drop_table('baselines')
    op.drop_index('ix_ml_models_created_at')
    op.drop_index('ix_ml_models_status')
    op.drop_table('ml_models')
    op.drop_index('ix_feature_sets_status')
    op.drop_table('feature_sets')
    op.drop_index('ix_datasets_created_at')
    op.drop_index('ix_datasets_status')
    op.drop_index('ix_datasets_name')
    op.drop_table('datasets')