/**
 * Training Page
 * Configure and run anomaly detection model training jobs with a step wizard.
 */
import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
    Card, Steps, Button, Select, Form, InputNumber, Slider,
    Table, Tag, Progress, Typography, Row, Col, Space, Divider,
    Radio, message, Alert, Popconfirm, Tabs, Modal, Input, Checkbox, Tooltip, Empty, Collapse, Badge
} from 'antd';
import {
    ExperimentOutlined, PlayCircleOutlined, CheckCircleOutlined,
    DatabaseOutlined, RocketOutlined,
    DeleteOutlined, MergeCellsOutlined
} from '@ant-design/icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { trainingService, Algorithm, Hyperparameter } from '@/services/trainingService';
import { datasetService, Dataset } from '@/services/datasetService';
import { featureService, FeatureSet } from '@/services/featureService';
import { FeatureAnalysisCard } from './components/FeatureAnalysisCard';
import { EyeOutlined, DownloadOutlined, RedoOutlined } from '@ant-design/icons';
import { useAuth } from '@/contexts/AuthContext';

const { Title, Text, Paragraph } = Typography;

type TrainingStage = 'binary' | 'anomaly_type';

const STAGE_CONFIG: Record<TrainingStage, {
    label: string;
    description: string;
    objective: string;
    evalMetric: string;
}> = {
    binary: {
        label: 'Stage 1: Binary anomaly detection',
        description: 'Normal records become label 0 and anomalous records become label 1.',
        objective: 'binary:logistic',
        evalMetric: 'logloss',
    },
    anomaly_type: {
        label: 'Stage 2: Anomaly-type classification',
        description: 'Normal records are excluded and existing anomaly-type classes are preserved.',
        objective: 'multi:softprob',
        evalMetric: 'mlogloss',
    },
};

/** Local time display timestamp (YYYY-MM-DD HH:mm) - storage still uses UTC internally */
function formatLocalTime(d: Date): string {
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    const hour = String(d.getHours()).padStart(2, '0');
    const minute = String(d.getMinutes()).padStart(2, '0');
    return `${year}-${month}-${day} ${hour}:${minute}`;
}

export function Training() {
    const [searchParams, setSearchParams] = useSearchParams();
    const navigate = useNavigate();
    const { hasRole } = useAuth();
    const canTrain = hasRole(['ADMIN', 'ML_ENGINEER']);

    // Initialize state from URL params to support refresh/bookmarks
    const [currentStep, setCurrentStep] = useState(Number(searchParams.get('step')) || 0);
    const [selectedDataset, setSelectedDataset] = useState<string | null>(searchParams.get('dataset') || null);
    const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('xgboost');
    const [trainingStage, setTrainingStage] = useState<TrainingStage>('binary');
    const [hyperparameters, setHyperparameters] = useState<Record<string, any>>({});
    const [imbalancedStrategy] = useState('class_weight');
    const [testSize, setTestSize] = useState<number>(0.2); // Default 20%
    const [featureConfig, setFeatureConfig] = useState<Record<string, boolean>>({
        transaction_features: true,
        behavioral_features: true,
        temporal_features: true,
        aggregation_features: true,
    });
    const [viewDataset, setViewDataset] = useState<Dataset | null>(null);
    const [viewModalOpen, setViewModalOpen] = useState(false);
    const [previewData, setPreviewData] = useState<{ columns: string[]; rows: unknown[]; total_rows: number } | null>(null);
    const [previewLoading, setPreviewLoading] = useState(false);
    const [selectedSplitJob, setSelectedSplitJob] = useState<string | null>(searchParams.get('split_job') || null);
    const [tuningMethod, setTuningMethod] = useState('manual');

    // Feature Engineering State
    const [featureSetId, setFeatureSetId] = useState<string | null>(searchParams.get('feature_set') || null);
    const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);

    // Merge state
    const [selectedMergeDatasets, setSelectedMergeDatasets] = useState<React.Key[]>([]);
    const [mergeModalOpen, setMergeModalOpen] = useState(false);
    const [mergeForm] = Form.useForm();
    const queryClient = useQueryClient();
    const [validationWarnings, setValidationWarnings] = useState<string[]>([]);
    const [warningModalOpen, setWarningModalOpen] = useState(false);
    const [jobRejected, setJobRejected] = useState(false);
    // Logistic Regression UX: only filter solver after user selects penalty
    const [lrPenaltyTouched, setLrPenaltyTouched] = useState(false);

    useEffect(() => {
        // Reset when switching algorithms or reloading defaults
        setLrPenaltyTouched(false);
    }, [selectedAlgorithm]);

    // Sync state to URL params
    useEffect(() => {
        const params: Record<string, string> = {};
        if (currentStep) params.step = String(currentStep);
        if (selectedDataset) params.dataset = selectedDataset;
        if (selectedSplitJob) params.split_job = selectedSplitJob;
        if (featureSetId) params.feature_set = featureSetId;

        // Only update if params changed to avoid infinite loops or noise
        // But setSearchParams is stable enough usually.
        // We use replace: true to avoid cluttering history for minor updates, 
        // but push for step changes? For simplicity, we'll just set it.
        setSearchParams(params, { replace: true });
    }, [currentStep, selectedDataset, selectedSplitJob, featureSetId, setSearchParams]);



    // Fetch algorithms
    const { data: algorithms, isLoading: algorithmsLoading, error: algorithmsError } = useQuery({
        queryKey: ['algorithms'],
        queryFn: () => trainingService.listAlgorithms(),
    });

    // Debugging: Log algorithms state
    useEffect(() => {
        console.log('🔍 Algorithms Debug:', {
            loading: algorithmsLoading,
            error: algorithmsError,
            data: algorithms,
            hasData: !!algorithms?.data,
            count: algorithms?.data?.length
        });

        if (algorithmsError) {
            message.error(`Failed to load algorithms: ${(algorithmsError as Error).message}`);
        }
    }, [algorithms, algorithmsLoading, algorithmsError]);

    // Fetch training jobs with auto-refresh for running jobs
    const { data: trainingJobs } = useQuery({
        queryKey: ['trainingJobs'],
        queryFn: () => trainingService.listJobs(),
        refetchInterval: (query) => {
            // Auto-refresh every 5 seconds if there are running jobs
            const hasRunningJobs = (query as any)?.state?.data?.data?.some(
                (job: any) => job.status === 'RUNNING' || job.status === 'QUEUED'
            );
            return hasRunningJobs ? 5000 : false;
        }
    });

    // Fetch datasets
    const { data: datasets, isLoading: datasetsLoading } = useQuery({
        queryKey: ['datasets', { includeMerged: true }],
        queryFn: () => datasetService.list(1, 100, undefined, true),
    });

    // Presets fetched conditionally

    // Merge datasets mutation
    const mergeDatasetMutation = useMutation({
        mutationFn: (values: { ids: string[]; name: string; description?: string }) =>
            datasetService.merge(values.ids, values.name, values.description),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['datasets'] });
            setMergeModalOpen(false);
            setSelectedMergeDatasets([]);
            mergeForm.resetFields();
            message.success('Datasets merged successfully');
        },
        onError: (error: any) => {
            const errorMsg = error.response?.data?.detail || error.message || 'Failed to merge datasets';
            message.error({
                content: (
                    <div style={{ whiteSpace: 'pre-wrap', textAlign: 'left' }}>
                        {errorMsg}
                    </div>
                ),
                duration: 5,
            });
        },
    });

    // Delete dataset mutation
    const deleteDatasetMutation = useMutation({
        mutationFn: (id: string) => datasetService.delete(id),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['datasets'] });
            message.success('Dataset deleted successfully');
        },
        onError: (err: any) => {
            const msg = err?.response?.data?.detail || err?.message || 'Failed to delete dataset';
            message.error(Array.isArray(msg) ? msg.join(', ') : msg);
        },
    });



    // Create training job mutation
    const createJobMutation = useMutation({
        mutationFn: trainingService.createJob,
        onSuccess: (response, variables) => {
            queryClient.invalidateQueries({ queryKey: ['trainingJobs'] });

            // Surface any hyperparameter conflict warnings BEFORE proceeding
            const warnings = (response as any).validation_warnings || [];
            const isRejected = (response as any).is_rejected || false;

            if (warnings.length > 0) {
                setValidationWarnings(warnings);
                setJobRejected(isRejected);
                setWarningModalOpen(true);
            }

            if ((variables as any).processing_only) {
                message.success('Data split task completed successfully');
                queryClient.invalidateQueries({ queryKey: ['datasets'] });
                if ((response as any)?.data?.id) {
                    setSelectedSplitJob((response as any).data.id);
                }
            } else if (!isRejected) {
                if (warnings.length === 0) {
                    message.success('Training job started!');
                } else {
                    message.warning('Training job queued — review hyperparameter warnings below.');
                }
                navigate('/jobs');
            } else {
                message.error('Job rejected due to invalid hyperparameters.');
            }
        },
        onError: (error: any) => {
            const detail = error?.response?.data?.detail;
            const msg = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.join(', ') : error?.message || 'Failed to start training';
            message.error(msg);
        },
    });

    // Delete training job mutation
    const deleteJobMutation = useMutation({
        mutationFn: trainingService.deleteJob,
        onSuccess: (_, jobId) => {
            queryClient.invalidateQueries({ queryKey: ['trainingJobs'] });
            if (selectedSplitJob === jobId) {
                setSelectedSplitJob(null);
            }
            message.success('Task deleted successfully');
        },
        onError: () => {
            message.error('Failed to delete task');
        },
    });

    // Generate Features Mutation (Step 2)
    const generateFeaturesMutation = useMutation({
        mutationFn: (data: any) => featureService.computeFeatures(data),
        onSuccess: (response: any) => {
            if (response.data.reused) {
                message.success(response.data.message || 'Feature set reused from registry.');
            } else {
                message.success('Feature generation started. Proceeding to analysis.');
            }
            setFeatureSetId(response.data.id);
            // Invalidate/refetch to show the new set in registry
            queryClient.invalidateQueries({ queryKey: ['featureSets'] });
        },
        onError: () => message.error('Failed to start feature generation')
    });

    // Resolve target dataset ID for feature engineering hooks
    const splitJob = trainingJobs?.data?.find((j: any) => j.id === selectedSplitJob);
    const targetDatasetId = splitJob?.metrics?.train_dataset_id || selectedDataset;

    // Fetch existing feature sets for registry
    const { data: featureSets_raw, refetch: refetchFeatureSets } = useQuery({
        queryKey: ['featureSets', targetDatasetId],
        queryFn: () => featureService.listFeatureSets(targetDatasetId),
        enabled: !!targetDatasetId && currentStep === 2,
    });
    const featureSets = featureSets_raw?.data || [];

    // Delete feature set mutation
    const deleteFeatureSetMutation = useMutation({
        mutationFn: (id: string) => featureService.deleteFeatureSet(id),
        onSuccess: () => {
            message.success('Feature set deleted');
            refetchFeatureSets();
            // If current selected set is deleted, deselect it
            if (featureSetId) {
                // Check if deleted id matches current. 
                // But mutation callback doesn't have id in closure easily unless passed.
                // We'll just let user handle it or add check if needed.
            }
        },
        onError: () => message.error('Failed to delete feature set')
    });

    // Update hyperparameters when algorithm changes
    useEffect(() => {
        if (algorithms?.data) {
            const algo = algorithms.data.find((a: Algorithm) => a.id === selectedAlgorithm);
            if (algo) {
                const defaults: Record<string, any> = {};
                algo.hyperparameters.forEach((hp: any) => {
                    defaults[hp.name] = hp.default;
                });
                defaults.training_stage = trainingStage;
                if (selectedAlgorithm === 'xgboost') {
                    defaults.objective = STAGE_CONFIG[trainingStage].objective;
                    defaults.eval_metric = STAGE_CONFIG[trainingStage].evalMetric;
                }
                setHyperparameters(defaults);
            }
        }
    }, [selectedAlgorithm, algorithms, trainingStage]);

    const handleView = async (record: Dataset) => {
        setViewDataset(record);
        setPreviewLoading(true);
        setViewModalOpen(true);
        try {
            const data = await datasetService.preview(record.id, 10);
            setPreviewData(data);
        } catch (error) {
            message.error('Failed to load preview');
        } finally {
            setPreviewLoading(false);
        }
    };

    const handleDownload = async () => {
        if (viewDataset) {
            try {
                message.loading({ content: 'Preparing download...', key: 'download' });
                const { download_url } = await datasetService.getDownloadUrl(viewDataset.id);

                // Try File System Access API
                // @ts-ignore
                if (window.showSaveFilePicker) {
                    try {
                        // @ts-ignore
                        const handle = await window.showSaveFilePicker({
                            suggestedName: `${viewDataset.name}.${viewDataset.file_format}`,
                            types: [{
                                description: 'Dataset File',
                                accept: {
                                    'text/csv': ['.csv'],
                                    'application/json': ['.json'],
                                    'application/octet-stream': ['.parquet']
                                }
                            }],
                        });
                        const writable = await handle.createWritable();
                        const response = await fetch(download_url);
                        await response.body?.pipeTo(writable);
                        message.success({ content: 'Download saved!', key: 'download' });
                        return;
                    } catch (err: any) {
                        if (err.name === 'AbortError') {
                            message.destroy('download');
                            return;
                        }
                        console.warn('File System Access API failed, falling back...', err);
                    }
                }

                // Fallback
                window.open(download_url, '_blank');
                message.success({ content: 'Download started!', key: 'download' });
            } catch (error) {
                console.error(error);
                message.error({ content: 'Failed to download', key: 'download' });
            }
        }
    };

    const steps = [
        { title: 'Select Data', icon: <DatabaseOutlined /> },
        { title: 'Data Split', icon: <MergeCellsOutlined /> },
        { title: 'Feature Eng', icon: <ExperimentOutlined /> },
        { title: 'Train', icon: <RocketOutlined /> },
    ];

    const handleStartTraining = async () => {
        // Resolve dataset ID from Split Job if not directly selected
        const splitJob = trainingJobs?.data?.find((j: any) => j.id === selectedSplitJob);
        const resolvedDatasetId = selectedDataset || splitJob?.metrics?.train_dataset_id;

        if (!resolvedDatasetId) {
            message.error('Please select a dataset or a split task');
            return;
        }
        if (!featureSetId) {
            message.error('Please generate features in Feature Engineering before training');
            return;
        }
        if (selectedAlgorithm !== 'xgboost') {
            message.error('Both AnomaLens training stages use XGBoost. Please select XGBoost.');
            return;
        }

        const stageConfig = STAGE_CONFIG[trainingStage];

        const payload = {
            name: `${selectedAlgorithm.toUpperCase()} Model ${formatLocalTime(new Date())}`,
            dataset_id: resolvedDatasetId,
            feature_set_id: featureSetId,
            feature_config: {
                ...featureConfig,
                selected_features: selectedFeatures.length > 0 ? selectedFeatures : undefined
            },
            algorithm: selectedAlgorithm,
            hyperparameters: {
                ...hyperparameters,
                // The worker reads this existing backend-compatible field to select
                // binary conversion or anomaly-type multiclass preparation.
                training_stage: trainingStage,
                objective: stageConfig.objective,
                eval_metric: stageConfig.evalMetric,
            },
            tuning_method: tuningMethod,
            tuning_config: {
                n_iter: tuningMethod === 'random' ? 20 : tuningMethod === 'bayesian' ? 30 : 10,
                cv_folds: 5,
            },
            imbalanced_strategy: imbalancedStrategy,
            test_size: testSize,
        };

        createJobMutation.mutate(payload);
    };

    const handleOpenMergeModal = () => {
        const selected = datasets?.data?.filter(d => selectedMergeDatasets.includes(d.id)) || [];
        if (selected.length > 0) {
            // Sort by name for consistency or use selection order? 
            // Default selection order might be random if relying on keys, but 'selected' is filtered from original list order
            // Enterprise naming: traceable default (Merge YYYY-MM-DD HH:mm or "Merge of A, B")
            const sourceList = selected.map(d => d.name).join(', ');
            const autoName = selected.length <= 2
                ? `Merge of ${sourceList}`
                : `Merge ${formatLocalTime(new Date())}`;

            mergeForm.setFieldsValue({
                name: autoName,
                description: `Merged dataset containing: ${sourceList}`
            });
        }
        setMergeModalOpen(true);
    };

    // Filter datasets
    const availableDatasets = datasets?.data?.filter((d: Dataset) => !d.parent_id) || [];
    const mergedDatasets = datasets?.data?.filter((d: Dataset) =>
        d.parent_id &&
        !d.name.includes('(Train Split)') &&
        !d.name.includes('(Test Split)')
    ) || [];
    // All datasets that can be split (single or merged) — used for adaptive Step 1
    const splittableDatasets = [...availableDatasets, ...mergedDatasets];

    const renderStepContent = () => {
        switch (currentStep) {
            case 0:

                return (
                    <div style={{ marginTop: 24 }}>
                        <Alert
                            message="Dataset Selection"
                            description={
                                availableDatasets.length === 1
                                    ? 'You can proceed directly to split your single dataset, or merge it with others if needed.'
                                    : availableDatasets.length > 1
                                        ? 'Select multiple datasets to merge them, or proceed to the next step to split individual or merged datasets.'
                                        : mergedDatasets.length > 0
                                            ? 'You have merged datasets. Proceed to the next step to split one for training.'
                                            : 'Upload datasets first to begin the training workflow.'
                            }
                            type="info"
                            showIcon
                            style={{ marginBottom: 24 }}
                        />

                        <Row gutter={24}>
                            {/* Left Col: Available Datasets */}
                            <Col span={24}>
                                <Card
                                    title={<Space><DatabaseOutlined /> Available Datasets</Space>}
                                    bordered={false}
                                    style={{ boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}
                                    extra={
                                        <Button
                                            type="primary"
                                            icon={<MergeCellsOutlined />}
                                            disabled={selectedMergeDatasets.length < 2}
                                            onClick={handleOpenMergeModal}
                                            style={{ backgroundColor: selectedMergeDatasets.length >= 2 ? '#1890ff' : undefined }}
                                        >
                                            Merge Selected ({selectedMergeDatasets.length})
                                        </Button>
                                    }
                                >
                                    <Table
                                        loading={datasetsLoading}
                                        dataSource={availableDatasets}
                                        rowKey="id"
                                        rowSelection={{
                                            selectedRowKeys: selectedMergeDatasets, // Enable multi-select for merging
                                            onChange: (keys) => setSelectedMergeDatasets(keys),
                                        }}
                                        columns={[
                                            {
                                                title: 'Dataset Name',
                                                dataIndex: 'name',
                                                key: 'name',
                                                render: (text: string) => <Text strong>{text}</Text>
                                            },
                                            {
                                                title: 'Rows',
                                                dataIndex: 'row_count',
                                                key: 'rows',
                                                width: 100,
                                                render: (count: number) => <Tag color="blue">{count?.toLocaleString()}</Tag>
                                            },
                                            {
                                                title: 'Cols',
                                                dataIndex: 'column_count',
                                                key: 'cols',
                                                width: 80,
                                                render: (count: number) => <Tag color="cyan">{count?.toLocaleString()}</Tag>
                                            },
                                            { title: 'Created', dataIndex: 'created_at', key: 'created', width: 100, render: (d: string) => new Date(d).toLocaleDateString() },
                                            {
                                                title: 'Action',
                                                key: 'action',
                                                width: 100,
                                                render: (_, record) => (
                                                    <Space>
                                                        <Tooltip title="Preview Dataset">
                                                            <Button icon={<EyeOutlined />} onClick={() => handleView(record)} />
                                                        </Tooltip>
                                                        <Popconfirm
                                                            title="Delete dataset?"
                                                            description="This action cannot be undone."
                                                            onConfirm={(e) => {
                                                                e?.stopPropagation();
                                                                deleteDatasetMutation.mutate(record.id);
                                                            }}
                                                            onCancel={(e) => e?.stopPropagation()}
                                                            okText="Yes"
                                                            cancelText="No"
                                                        >
                                                            <Tooltip title="Delete Dataset">
                                                                <Button danger icon={<DeleteOutlined />} onClick={(e) => e.stopPropagation()} />
                                                            </Tooltip>
                                                        </Popconfirm>
                                                    </Space>
                                                )
                                            }
                                        ]}
                                        pagination={{ pageSize: 5 }}
                                        size="small"
                                    />
                                </Card>
                            </Col>

                            {/* Right Col: Merged Datasets */}
                            <Col span={24} style={{ marginTop: 24 }}>
                                <Card
                                    title={<Space><MergeCellsOutlined /> Merged Datasets</Space>}
                                    bordered={false}
                                    style={{ boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}
                                >
                                    <Table
                                        loading={datasetsLoading}
                                        dataSource={mergedDatasets}
                                        rowKey="id"
                                        columns={[
                                            {
                                                title: 'Merged Name',
                                                dataIndex: 'name',
                                                key: 'name',
                                                render: (text: string) => <Text strong>{text}</Text>
                                            },
                                            {
                                                title: 'Rows',
                                                dataIndex: 'row_count',
                                                key: 'rows',
                                                width: 100,
                                                render: (count: number) => <Tag color="purple">{count?.toLocaleString()}</Tag>
                                            },
                                            {
                                                title: 'Cols',
                                                dataIndex: 'column_count',
                                                key: 'cols',
                                                width: 80,
                                                render: (count: number) => <Tag color="cyan">{count?.toLocaleString()}</Tag>
                                            },
                                            { title: 'Created', dataIndex: 'created_at', key: 'created', width: 100, render: (d: string) => new Date(d).toLocaleDateString() },
                                            {
                                                title: 'Action',
                                                key: 'action',
                                                width: 100,
                                                render: (_, record) => (
                                                    <Space>
                                                        <Tooltip title="Preview Dataset">
                                                            <Button icon={<EyeOutlined />} onClick={() => handleView(record)} />
                                                        </Tooltip>
                                                        <Popconfirm
                                                            title="Delete dataset?"
                                                            description="This action cannot be undone."
                                                            onConfirm={(e) => {
                                                                e?.stopPropagation();
                                                                deleteDatasetMutation.mutate(record.id);
                                                            }}
                                                            onCancel={(e) => e?.stopPropagation()}
                                                            okText="Yes"
                                                            cancelText="No"
                                                        >
                                                            <Tooltip title="Delete Dataset">
                                                                <Button danger icon={<DeleteOutlined />} onClick={(e) => e.stopPropagation()} />
                                                            </Tooltip>
                                                        </Popconfirm>
                                                    </Space>
                                                )
                                            }
                                        ]}
                                        pagination={{ pageSize: 5 }}
                                        size="small"
                                        locale={{ emptyText: 'No merged datasets. Select & Merge from left.' }}
                                    />
                                </Card>
                            </Col>
                        </Row>
                    </div>
                );

            case 1:
                return (
                    <div style={{ marginTop: 24 }}>
                        <Alert
                            message="Train-Test Split"
                            description={
                                splittableDatasets.length === 0
                                    ? 'No datasets available. Please upload or merge datasets in the previous step.'
                                    : availableDatasets.length > 0 && mergedDatasets.length > 0
                                        ? 'Select any dataset (single or merged) to split for training and validation.'
                                        : availableDatasets.length > 0
                                            ? 'Select a dataset to split for training and validation.'
                                            : 'Select a merged dataset to split for training and validation.'
                            }
                            type={splittableDatasets.length === 0 ? 'warning' : 'info'}
                            showIcon
                            style={{ marginBottom: 24 }}
                        />

                        <Card
                            title={<Space><DatabaseOutlined /> Select Dataset to Split</Space>}
                            style={{ marginBottom: 24, boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}
                        >
                            <Table
                                loading={datasetsLoading}
                                dataSource={splittableDatasets}
                                rowKey="id"
                                rowSelection={{
                                    type: 'radio',
                                    selectedRowKeys: selectedDataset ? [selectedDataset] : [],
                                    onChange: (keys: React.Key[]) => {
                                        setSelectedDataset(keys[0] as string);
                                    },
                                }}
                                columns={[
                                    {
                                        title: 'Dataset Name',
                                        dataIndex: 'name',
                                        key: 'name',
                                        render: (text: string, record: Dataset) => (
                                            <Space>
                                                <Text strong>{text}</Text>
                                                {record.parent_id ? <Tag color="purple">Merged</Tag> : <Tag color="blue">Single</Tag>}
                                            </Space>
                                        )
                                    },
                                    {
                                        title: 'Rows',
                                        dataIndex: 'row_count',
                                        key: 'rows',
                                        width: 100,
                                        render: (count: number) => <Tag color="blue">{count?.toLocaleString()}</Tag>
                                    },
                                    {
                                        title: 'Cols',
                                        dataIndex: 'column_count',
                                        key: 'cols',
                                        width: 80,
                                        render: (count: number) => <Tag color="cyan">{count?.toLocaleString()}</Tag>
                                    },
                                    { title: 'Created', dataIndex: 'created_at', key: 'created', width: 150, render: (d: string) => new Date(d).toLocaleDateString() },
                                    {
                                        title: 'Action',
                                        key: 'action',
                                        width: 80,
                                        render: (_: unknown, record: Dataset) => (
                                            <Tooltip title="Preview Dataset">
                                                <Button icon={<EyeOutlined />} onClick={() => handleView(record)} />
                                            </Tooltip>
                                        )
                                    }
                                ]}
                                pagination={{ pageSize: 5 }}
                                size="small"
                                locale={{ emptyText: 'No datasets available. Go back to upload or merge data.' }}
                            />
                        </Card>

                        {selectedDataset && (
                            <Card title="Data Slicer Configuration" className="fade-in">
                                <Row gutter={24} align="middle">
                                    <Col span={12}>
                                        <Title level={5}>Test Split Ratio</Title>
                                        <Text type="secondary">
                                            Adjust the slider to reserve data for testing.
                                            Higher test size = more confident validation, but less data for training.
                                        </Text>
                                        <div style={{ marginTop: 16 }}>
                                            <Slider
                                                min={0.1}
                                                max={0.5}
                                                step={0.05}
                                                value={testSize}
                                                onChange={setTestSize}
                                                marks={{
                                                    0.1: '10%',
                                                    0.2: '20% (Default)',
                                                    0.3: '30%',
                                                    0.5: '50%'
                                                }}
                                                tooltip={{ formatter: (value) => `${(value! * 100).toFixed(0)}% Testing` }}
                                            />
                                        </div>
                                    </Col>
                                    <Col span={12}>
                                        <Card type="inner" title="Projected Data Split" size="small">
                                            {selectedDataset ? (() => {
                                                let totalRows = 0;
                                                let sourceName = "";

                                                const ds = datasets?.data?.find((d: any) => d.id === selectedDataset);
                                                totalRows = ds?.row_count || 0;
                                                sourceName = ds ? `Using dataset: ${ds.name}` : '';

                                                return (
                                                    <div style={{ textAlign: 'center' }}>
                                                        <Title level={4}>
                                                            {totalRows ?
                                                                `${totalRows.toLocaleString()} rows` :
                                                                "Unknown Size"}
                                                        </Title>

                                                        <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginTop: 10 }}>
                                                            <Tag color="geekblue" style={{ fontSize: '14px', padding: '5px 10px' }}>
                                                                🟦 Training: {(totalRows * (1 - testSize)).toFixed(0)}
                                                            </Tag>
                                                            <Tag color="orange" style={{ fontSize: '14px', padding: '5px 10px' }}>
                                                                🟧 Testing: {(totalRows * testSize).toFixed(0)}
                                                            </Tag>
                                                        </div>
                                                        <Progress
                                                            percent={100 - testSize * 100}
                                                            success={{ percent: 0 }}
                                                            strokeColor={{ '0%': '#2f54eb', '100%': '#2f54eb' }}
                                                            trailColor="#fa8c16"
                                                            showInfo={false}
                                                            style={{ marginTop: 15 }}
                                                        />
                                                        <div style={{ marginTop: 8, fontSize: '12px', color: '#888' }}>
                                                            {sourceName}
                                                        </div>
                                                    </div>
                                                );
                                            })() : (
                                                <div style={{ textAlign: 'center', padding: '20px', color: '#999' }}>
                                                    Select a Dataset to see row counts.
                                                </div>
                                            )}
                                        </Card>
                                    </Col>
                                </Row>

                                <div style={{ marginTop: 24, display: 'flex', justifyContent: 'flex-end' }}>
                                    <Button
                                        type="primary"
                                        icon={<MergeCellsOutlined />}
                                        loading={createJobMutation.isPending}
                                        onClick={() => {
                                            if (!selectedDataset) return message.error('Please select a dataset');
                                            createJobMutation.mutate({
                                                name: `Split ${formatLocalTime(new Date())}`,
                                                dataset_id: selectedDataset,
                                                feature_config: featureConfig,
                                                algorithm: selectedAlgorithm,
                                                hyperparameters: {},
                                                imbalanced_strategy: 'class_weight',
                                                test_size: testSize,
                                                processing_only: true
                                            });
                                        }}
                                    >
                                        Split & Save to Cloud
                                    </Button>
                                </div>
                            </Card>
                        )}

                        {/* Recent Split Tasks Display */}
                        <div style={{ marginTop: 24 }}>
                            <Card title="Recent Data Prep Tasks" size="small">
                                <Table
                                    dataSource={trainingJobs?.data?.filter((j: any) => j.processing_only || j.status === 'DATA_PREPARED') || []}
                                    rowKey="id"
                                    size="small"
                                    pagination={{ pageSize: 3 }}
                                    rowSelection={{
                                        type: 'radio',
                                        selectedRowKeys: selectedSplitJob ? [selectedSplitJob] : [],
                                        onChange: (keys) => {
                                            setSelectedSplitJob(keys[0] as string);
                                            setFeatureSetId(null);
                                        }
                                    }}
                                    columns={[
                                        { title: 'Job Name', dataIndex: 'name', key: 'name' },
                                        { title: 'Dataset ID', dataIndex: 'dataset_id', key: 'ds', ellipsis: true },
                                        { title: 'Status', dataIndex: 'status', key: 'status', render: (s: string) => <Tag color="green">{s}</Tag> },
                                        { title: 'Created', dataIndex: 'created_at', key: 'created', render: (d: string) => new Date(d).toLocaleString() },
                                        {
                                            title: 'Action',
                                            key: 'action',
                                            render: (_, record: any) => (
                                                <Popconfirm
                                                    title="Delete task?"
                                                    description="This cannot be undone."
                                                    onConfirm={(e) => {
                                                        e?.stopPropagation();
                                                        deleteJobMutation.mutate(record.id);
                                                    }}
                                                    onCancel={(e) => e?.stopPropagation()}
                                                    okText="Yes"
                                                    cancelText="No"
                                                    disabled={!canTrain}
                                                >
                                                    <Tooltip title={!canTrain ? "Your role does not have permission to delete tasks." : ""}>
                                                        <Button
                                                            danger
                                                            icon={<DeleteOutlined />}
                                                            size="small"
                                                            onClick={(e) => e.stopPropagation()}
                                                            disabled={!canTrain}
                                                        />
                                                    </Tooltip>
                                                </Popconfirm>
                                            )
                                        }
                                    ]}
                                    locale={{ emptyText: 'No split tasks running' }}
                                />
                            </Card>
                        </div>
                    </div>
                );

            case 2:
                return (
                    <div style={{ marginTop: 24 }}>
                        <Alert
                            message="Feature Engineering"
                            description="Select which feature groups to generate. These features will be computed on the split data."
                            type="info"
                            showIcon
                            style={{ marginBottom: 24 }}
                        />
                        {selectedSplitJob && (
                            <Alert
                                message={`Using Data Split: ${trainingJobs?.data?.find((j: any) => j.id === selectedSplitJob)?.name || 'Unknown'}`}
                                description={(() => {
                                    const job = trainingJobs?.data?.find((j: any) => j.id === selectedSplitJob);
                                    if (job?.metrics?.train_dataset_id) {
                                        return <Text type="success">Verified: Using split dataset ({job.metrics.train_rows?.toLocaleString()} rows)</Text>;
                                    }
                                    return <Text type="warning">Warning: Split dataset not found. Using raw dataset.</Text>;
                                })()}
                                type="success"
                                showIcon
                                style={{ marginBottom: 16 }}
                            />

                        )}

                        {/* Feature Set Registry */}
                        <Card
                            title={<Space><DatabaseOutlined /> Feature Set Registry</Space>}
                            size="small"
                            style={{ marginBottom: 24, background: '#f6ffed', borderColor: '#b7eb8f' }}
                            extra={<Tag color="success">Reuse Existing Features</Tag>}
                        >
                            <Table
                                dataSource={featureSets}
                                rowKey="id"
                                size="small"
                                pagination={{ pageSize: 3 }}
                                locale={{ emptyText: 'No feature sets found. Configure below to generate one.' }}
                                columns={[
                                    {
                                        title: 'Feature Set Name',
                                        dataIndex: 'name',
                                        key: 'name',
                                        render: (text: string, record: FeatureSet) => (
                                            <Space direction="vertical" size={0}>
                                                <Text strong>{text}</Text>
                                                <Text type="secondary" style={{ fontSize: 11 }}>
                                                    {new Date(record.created_at).toLocaleString()}
                                                </Text>
                                            </Space>
                                        )
                                    },
                                    {
                                        title: 'Features',
                                        dataIndex: 'feature_count',
                                        key: 'count',
                                        render: (count: number) => count ? <Tag color="blue">{count}</Tag> : <Tag>Pending</Tag>
                                    },
                                    {
                                        title: 'Status',
                                        dataIndex: 'status',
                                        key: 'status',
                                        render: (status: string) => (
                                            <Tag color={status === 'COMPLETED' ? 'green' : status === 'FAILED' ? 'red' : 'processing'}>
                                                {status}
                                            </Tag>
                                        )
                                    },
                                    {
                                        title: 'Action',
                                        key: 'action',
                                        render: (_, record: FeatureSet) => (
                                            <Space>
                                                <Tooltip title="Use this Feature Set">
                                                    <Button
                                                        type="primary"
                                                        ghost
                                                        icon={<CheckCircleOutlined />}
                                                        disabled={record.status !== 'COMPLETED'}
                                                        onClick={() => setFeatureSetId(record.id)}
                                                    >
                                                        Select
                                                    </Button>
                                                </Tooltip>
                                                <Popconfirm
                                                    title="Delete feature set?"
                                                    onConfirm={() => deleteFeatureSetMutation.mutate(record.id)}
                                                    okText="Yes"
                                                    cancelText="No"
                                                >
                                                    <Button danger icon={<DeleteOutlined />} size="small" />
                                                </Popconfirm>
                                            </Space>
                                        )
                                    }
                                ]}
                            />
                        </Card>

                        {!featureSetId ? (
                            <Card title="Feature Engineering Strategy">
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                                    <Checkbox
                                        checked={featureConfig.transaction_features}
                                        onChange={(e) => setFeatureConfig(prev => ({ ...prev, transaction_features: e.target.checked }))}
                                    >
                                        <Text strong>Record Features</Text> <br />
                                        <Text type="secondary">Transformations and derived features for anomaly detection</Text>
                                    </Checkbox>
                                    <Checkbox
                                        checked={featureConfig.behavioral_features}
                                        onChange={(e) => setFeatureConfig(prev => ({ ...prev, behavioral_features: e.target.checked }))}
                                    >
                                        <Text strong>Behavioral Features</Text> <br />
                                        <Text type="secondary">Patterns, frequency, velocity, and usage behavior indicators</Text>
                                    </Checkbox>
                                    <Checkbox
                                        checked={featureConfig.temporal_features}
                                        onChange={(e) => setFeatureConfig(prev => ({ ...prev, temporal_features: e.target.checked }))}
                                    >
                                        <Text strong>Temporal Features</Text> <br />
                                        <Text type="secondary">Time of day, day of week, seasonality indicators</Text>
                                    </Checkbox>
                                    <Checkbox
                                        checked={featureConfig.aggregation_features}
                                        onChange={(e) => setFeatureConfig(prev => ({ ...prev, aggregation_features: e.target.checked }))}
                                    >
                                        <Text strong>Aggregation Features</Text> <br />
                                        <Text type="secondary">Aggregations (Sum, Avg, Count) over time windows (1h, 24h, 7d)</Text>
                                    </Checkbox>

                                    <Divider />
                                    <Button
                                        type="primary"
                                        icon={<ExperimentOutlined />}
                                        size="large"
                                        onClick={() => {
                                            // Resolver correct dataset ID (Split > Raw)
                                            const splitJob = trainingJobs?.data?.find((j: any) => j.id === selectedSplitJob);
                                            const targetDatasetId = splitJob?.metrics?.train_dataset_id || selectedDataset;

                                            if (!targetDatasetId) return message.error('No dataset selected. Please select a split job or dataset.');

                                            generateFeaturesMutation.mutate({
                                                dataset_id: targetDatasetId,
                                                name: `Features ${formatLocalTime(new Date())}`,
                                                ...featureConfig
                                            });
                                        }}
                                        loading={generateFeaturesMutation.isPending}
                                    >
                                        Generate & Analyze Features
                                    </Button>
                                </div>
                            </Card>
                        ) : (
                            <>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                                    <Text type="secondary" style={{ fontSize: 13 }}>
                                        Enabled: {[
                                            featureConfig.transaction_features && 'Record',
                                            featureConfig.behavioral_features && 'Behavioral',
                                            featureConfig.temporal_features && 'Temporal',
                                            featureConfig.aggregation_features && 'Aggregation',
                                        ].filter(Boolean).join(', ') || 'None'}
                                    </Text>
                                    <Button
                                        icon={<RedoOutlined />}
                                        onClick={() => {
                                            Modal.confirm({
                                                title: 'Reconfigure Features?',
                                                content: 'This will delete the current feature set and let you re-run feature engineering with a new configuration. This action cannot be undone.',
                                                okText: 'Yes, Reconfigure',
                                                okType: 'danger',
                                                cancelText: 'Cancel',
                                                onOk: async () => {
                                                    try {
                                                        await featureService.deleteFeatureSet(featureSetId!);
                                                        setFeatureSetId(null);
                                                        setSelectedFeatures([]);
                                                        message.success('Feature set deleted. You can now reconfigure and re-run.');
                                                    } catch {
                                                        message.error('Failed to delete feature set. Please try again.');
                                                    }
                                                },
                                            });
                                        }}
                                    >
                                        Reconfigure Features
                                    </Button>
                                </div>
                                <FeatureAnalysisCard
                                    featureSetId={featureSetId}
                                    datasetId={(() => {
                                        const splitJob = trainingJobs?.data?.find((j: any) => j.id === selectedSplitJob);
                                        return splitJob?.metrics?.train_dataset_id || selectedDataset!;
                                    })()}
                                    onSelectionChange={setSelectedFeatures}
                                />
                            </>
                        )
                        }
                    </div >
                );

            case 3:
                const selectedAlgo = algorithms?.data?.find((a: Algorithm) => a.id === selectedAlgorithm);
                return (
                    <div style={{ marginTop: 24 }}>
                        <Row gutter={24}>
                            <Col span={7}>
                                <Card title="Model Configuration">
                                    <Title level={5}>Training Stage</Title>
                                    <Radio.Group
                                        value={trainingStage}
                                        onChange={(e) => {
                                            setTrainingStage(e.target.value as TrainingStage);
                                            setSelectedAlgorithm('xgboost');
                                        }}
                                        style={{ width: '100%' }}
                                    >
                                        <Space direction="vertical" style={{ width: '100%' }}>
                                            {(Object.keys(STAGE_CONFIG) as TrainingStage[]).map((stage) => (
                                                <Radio.Button
                                                    key={stage}
                                                    value={stage}
                                                    style={{ width: '100%', height: 'auto', padding: 12 }}
                                                >
                                                    <Text strong>{STAGE_CONFIG[stage].label}</Text>
                                                    <br />
                                                    <Text type="secondary" style={{ fontSize: 12 }}>
                                                        {STAGE_CONFIG[stage].description}
                                                    </Text>
                                                </Radio.Button>
                                            ))}
                                        </Space>
                                    </Radio.Group>
                                    <Alert
                                        message={`XGBoost ${trainingStage === 'binary' ? 'binary' : 'multiclass'} configuration`}
                                        description={`Objective: ${STAGE_CONFIG[trainingStage].objective} | Metric: ${STAGE_CONFIG[trainingStage].evalMetric}`}
                                        type="info"
                                        showIcon
                                        style={{ marginTop: 16 }}
                                    />
                                    <Divider />
                                    <Title level={5}>Algorithm</Title>
                                    {algorithmsLoading ? (
                                        <div style={{ textAlign: 'center', padding: '40px 0' }}>
                                            <Space direction="vertical">
                                                <div className="spinner" />
                                                <Text type="secondary">Loading algorithms...</Text>
                                            </Space>
                                        </div>
                                    ) : algorithmsError ? (
                                        <Alert
                                            message="Error Loading Algorithms"
                                            description={(algorithmsError as Error).message}
                                            type="error"
                                            showIcon
                                        />
                                    ) : !algorithms?.data || algorithms.data.length === 0 ? (
                                        <Empty description="No algorithms available" />
                                    ) : (
                                        <Radio.Group
                                            value={selectedAlgorithm}
                                            onChange={(e) => setSelectedAlgorithm(e.target.value)}
                                            style={{ width: '100%' }}
                                        >
                                            <Space direction="vertical" style={{ width: '100%' }}>
                                                {algorithms.data.map((algo: Algorithm) => (
                                                    <Radio.Button
                                                        key={algo.id}
                                                        value={algo.id}
                                                        disabled={algo.id !== 'xgboost'}
                                                        style={{ width: '100%', height: 'auto', padding: 12 }}
                                                    >
                                                        <Text strong>{algo.name}</Text>
                                                    </Radio.Button>
                                                ))}
                                            </Space>
                                        </Radio.Group>
                                    )}

                                    <Divider />
                                    <Tooltip title={!canTrain ? "Your role does not have permission to train models." : ""}>
                                        <Button
                                            type="primary"
                                            size="large"
                                            icon={<PlayCircleOutlined />}
                                            onClick={handleStartTraining}
                                            loading={createJobMutation.isPending}
                                            style={{ width: '100%' }}
                                            disabled={!canTrain}
                                        >
                                            Start Training
                                        </Button>
                                    </Tooltip>
                                </Card>
                            </Col>
                            <Col span={17}>
                                <Card
                                    title="Tuning Method"
                                    style={{ marginBottom: 16 }}
                                >
                                    <Radio.Group
                                        value={tuningMethod}
                                        onChange={(e) => setTuningMethod(e.target.value)}
                                        style={{ width: '100%' }}
                                        buttonStyle="solid"
                                    >
                                        <Row gutter={12}>
                                            <Col span={6}>
                                                <Radio.Button value="manual" style={{ width: '100%', height: 'auto', padding: '12px 8px', textAlign: 'center' }}>
                                                    <Text strong style={{ fontSize: 15 }}>Manual</Text>
                                                    <br />
                                                    <Text type="secondary" style={{ fontSize: 12 }}>
                                                        Set manually
                                                    </Text>
                                                </Radio.Button>
                                            </Col>
                                            <Col span={6}>
                                                <Radio.Button value="grid" style={{ width: '100%', height: 'auto', padding: '12px 8px', textAlign: 'center' }}>
                                                    <Text strong style={{ fontSize: 15 }}>Grid Search</Text>
                                                    <br />
                                                    <Text type="secondary" style={{ fontSize: 12 }}>
                                                        Exhaustive search
                                                    </Text>
                                                </Radio.Button>
                                            </Col>
                                            <Col span={6}>
                                                <Radio.Button value="random" style={{ width: '100%', height: 'auto', padding: '12px 8px', textAlign: 'center' }}>
                                                    <Text strong style={{ fontSize: 15 }}>Random Search</Text>
                                                    <br />
                                                    <Text type="secondary" style={{ fontSize: 12 }}>
                                                        Random sampling
                                                    </Text>
                                                </Radio.Button>
                                            </Col>
                                            <Col span={6}>
                                                <Radio.Button value="bayesian" style={{ width: '100%', height: 'auto', padding: '12px 8px', textAlign: 'center' }}>
                                                    <Text strong style={{ fontSize: 15 }}>Bayesian Opt.</Text>
                                                    <br />
                                                    <Text type="secondary" style={{ fontSize: 12 }}>
                                                        Smart search
                                                    </Text>
                                                </Radio.Button>
                                            </Col>
                                        </Row>
                                    </Radio.Group>

                                    {/* Visual Feedback for Selected Method */}
                                    {tuningMethod !== 'manual' && (
                                        <Alert
                                            message={`${tuningMethod === 'grid' ? 'Grid Search' : tuningMethod === 'random' ? 'Random Search' : 'Bayesian Optimization'} Selected`}
                                            description={
                                                tuningMethod === 'grid'
                                                    ? 'Will search all parameter combinations exhaustively'
                                                    : tuningMethod === 'random'
                                                        ? `Will randomly sample ${20} parameter combinations`
                                                        : `Will intelligently search ${30} parameter combinations using Bayesian optimization`
                                            }
                                            type="info"
                                            showIcon
                                            style={{ marginTop: 16 }}
                                        />
                                    )}
                                </Card>

                                <Card
                                    title="Parameter Values"
                                >
                                    {tuningMethod === 'manual' ? (
                                        // ── Manual Mode: Grouped, full-featured hyperparameter controls ──
                                        (() => {
                                            const hps: Hyperparameter[] = selectedAlgo?.hyperparameters || [];

                                            // ── Logistic Regression constraints (frontend guard rails) ───────
                                            const isLR = selectedAlgo?.id === 'logistic_regression';
                                            const LR_SOLVER_PENALTIES: Record<string, Array<string | null>> = {
                                                'lbfgs': ['l2', null],
                                                'liblinear': ['l1', 'l2'],
                                                'newton-cg': ['l2', null],
                                                'sag': ['l2', null],
                                                'saga': ['l1', 'l2', 'elasticnet', null],
                                            };

                                            const normLR = (v: any): any => {
                                                if (typeof v === 'string') {
                                                    const s = v.trim().toLowerCase();
                                                    if (s === 'none') return null;
                                                    if (s === 'true') return true;
                                                    if (s === 'false') return false;
                                                }
                                                return v;
                                            };

                                            const getLRVal = (name: string, fallback: any) => {
                                                const v = hyperparameters[name] !== undefined ? hyperparameters[name] : fallback;
                                                return normLR(v);
                                            };

                                            const lrSolver = String(getLRVal('solver', 'lbfgs'));
                                            const lrPenalty = getLRVal('penalty', 'l2') as string | null;
                                            const lrAllowedSolvers = Object.keys(LR_SOLVER_PENALTIES).filter(s => (LR_SOLVER_PENALTIES[s] ?? []).includes(lrPenalty));
                                            const lrDualAllowed = lrSolver === 'liblinear' && lrPenalty === 'l2';
                                            const lrL1RatioAllowed = lrSolver === 'saga' && lrPenalty === 'elasticnet';


                                            // Collect unique groups in declaration order
                                            const groupOrder: string[] = [];
                                            const grouped: Record<string, Hyperparameter[]> = {};
                                            hps.forEach((hp) => {
                                                const g = hp.group || 'General';
                                                if (!grouped[g]) {
                                                    grouped[g] = [];
                                                    groupOrder.push(g);
                                                }
                                                grouped[g].push(hp);
                                            });

                                            const renderControl = (hp: Hyperparameter) => {
                                                const currentVal = hyperparameters[hp.name] !== undefined
                                                    ? hyperparameters[hp.name]
                                                    : hp.default;

                                                const setVal = (val: any) =>
                                                    setHyperparameters(prev => ({ ...prev, [hp.name]: val }));

                                                if (hp.type === 'select') {
                                                    // Logistic Regression: solver↔penalty filtering + dual gating
                                                    if (isLR && (hp.name === 'solver' || hp.name === 'penalty' || hp.name === 'dual')) {
                                                        let options = hp.options?.map(o => ({ label: o.label, value: o.value })) || [];

                                                        if (hp.name === 'penalty') {
                                                            // Show ALL penalties first; only after the user selects a penalty
                                                            // do we narrow the solver list.
                                                            // (We still auto-correct solver after penalty is selected.)
                                                        }

                                                        if (hp.name === 'solver') {
                                                            if (lrPenaltyTouched) {
                                                                const allowed = new Set(lrAllowedSolvers.length ? lrAllowedSolvers : Object.keys(LR_SOLVER_PENALTIES));
                                                                options = options.filter(o => allowed.has(String(o.value)));
                                                            }
                                                        }

                                                        if (hp.name === 'dual') {
                                                            options = options.map(o => {
                                                                if (String(o.value).toLowerCase() === 'true' && !lrDualAllowed) {
                                                                    return { ...o, disabled: true, label: `${o.label} (requires solver=liblinear & penalty=l2)` } as any;
                                                                }
                                                                return o as any;
                                                            });
                                                        }

                                                        const onChange = (val: any) => {
                                                            const v = normLR(val);
                                                            setHyperparameters(prev => {
                                                                const next: Record<string, any> = { ...prev, [hp.name]: val };

                                                                if (hp.name === 'solver') {
                                                                    const newSolver = String(v);
                                                                    const allowedPen = LR_SOLVER_PENALTIES[newSolver] ?? ['l2', null];
                                                                    const existingPenalty = normLR(prev['penalty'] ?? 'l2') as string | null;
                                                                    if (!allowedPen.includes(existingPenalty)) {
                                                                        next['penalty'] = allowedPen[0] === null ? 'None' : allowedPen[0];
                                                                    }

                                                                    const newDualAllowed = newSolver === 'liblinear' && normLR(next['penalty']) === 'l2';
                                                                    if (!newDualAllowed && String(normLR(prev['dual'] ?? 'False')).toLowerCase() === 'true') {
                                                                        next['dual'] = 'False';
                                                                    }

                                                                    const newL1Allowed = newSolver === 'saga' && normLR(next['penalty']) === 'elasticnet';
                                                                    if (!newL1Allowed) delete next['l1_ratio'];
                                                                }

                                                                if (hp.name === 'penalty') {
                                                                    // User explicitly chose a penalty → start filtering solver list
                                                                    setLrPenaltyTouched(true);
                                                                    const newPenalty = v as string | null;
                                                                    const allowedSolvers = Object.keys(LR_SOLVER_PENALTIES).filter(s => (LR_SOLVER_PENALTIES[s] ?? []).includes(newPenalty));
                                                                    const existingSolver = String(normLR(prev['solver'] ?? 'lbfgs'));
                                                                    if (!allowedSolvers.includes(existingSolver)) {
                                                                        next['solver'] = allowedSolvers[0] ?? 'lbfgs';
                                                                    }

                                                                    const newDualAllowed = String(normLR(next['solver'])) === 'liblinear' && newPenalty === 'l2';
                                                                    if (!newDualAllowed && String(normLR(prev['dual'] ?? 'False')).toLowerCase() === 'true') {
                                                                        next['dual'] = 'False';
                                                                    }

                                                                    const newL1Allowed = String(normLR(next['solver'])) === 'saga' && newPenalty === 'elasticnet';
                                                                    if (!newL1Allowed) delete next['l1_ratio'];
                                                                }

                                                                if (hp.name === 'dual') {
                                                                    if (String(v).toLowerCase() === 'true' && !lrDualAllowed) {
                                                                        next['dual'] = 'False';
                                                                    }
                                                                }

                                                                return next;
                                                            });
                                                        };

                                                        return (
                                                            <Select
                                                                value={currentVal}
                                                                onChange={onChange}
                                                                style={{ width: '100%' }}
                                                                options={options as any}
                                                            />
                                                        );
                                                    }
                                                    return (
                                                        <Select
                                                            value={currentVal}
                                                            onChange={setVal}
                                                            style={{ width: '100%' }}
                                                            options={hp.options?.map(o => ({ label: o.label, value: o.value }))}
                                                        />
                                                    );
                                                }

                                                // Numeric (int / float)
                                                const step = hp.type === 'float' ? 0.01 : 1;
                                                const minV = hp.min ?? 0;
                                                const maxV = hp.max ?? 100;
                                                const precision = hp.type === 'float' ? 3 : 0;

                                                // Build marks: min, default, max
                                                const marks: Record<number, any> = {};
                                                marks[minV] = { style: { fontSize: 11 }, label: hp.type === 'float' ? minV.toFixed(2) : minV };
                                                if (typeof hp.default === 'number' && hp.default > minV && hp.default < maxV) {
                                                    marks[hp.default] = { style: { fontSize: 11, color: '#1677ff' }, label: `${hp.default}` };
                                                }
                                                marks[maxV] = { style: { fontSize: 11 }, label: hp.type === 'float' ? maxV.toFixed(2) : maxV };

                                                // Logistic Regression: l1_ratio is only valid when solver='saga' and penalty='elasticnet'
                                                const disableNumeric = isLR && hp.name === 'l1_ratio' && !lrL1RatioAllowed;

                                                return (
                                                    <Row gutter={12} align="middle">
                                                        <Col flex="auto">
                                                            <Slider
                                                                min={minV}
                                                                max={maxV}
                                                                step={step}
                                                                value={typeof currentVal === 'number' ? currentVal : Number(currentVal)}
                                                                onChange={setVal}
                                                                marks={marks}
                                                                disabled={disableNumeric}
                                                                tooltip={{ formatter: (v) => v !== undefined ? (hp.type === 'float' ? Number(v).toFixed(3) : v) : '' }}
                                                            />
                                                        </Col>
                                                        <Col flex="90px">
                                                            <InputNumber
                                                                min={minV}
                                                                max={maxV}
                                                                step={step}
                                                                precision={precision}
                                                                value={typeof currentVal === 'number' ? currentVal : Number(currentVal)}
                                                                onChange={(v) => setVal(v ?? hp.default)}
                                                                style={{ width: '100%' }}
                                                                size="small"
                                                                disabled={disableNumeric}
                                                            />
                                                        </Col>
                                                    </Row>
                                                );
                                            };

                                            return (
                                                <Collapse
                                                    defaultActiveKey={groupOrder}
                                                    bordered={false}
                                                    style={{ background: 'transparent' }}
                                                    items={groupOrder.map(group => ({
                                                        key: group,
                                                        label: (
                                                            <Space>
                                                                <Text strong style={{ fontSize: 14 }}>{group}</Text>
                                                                <Badge count={grouped[group].length} style={{ backgroundColor: '#f0f0f0', color: '#666', fontSize: 11 }} />
                                                            </Space>
                                                        ),
                                                        children: (
                                                            <div style={{ paddingLeft: 4 }}>
                                                                {grouped[group].map(hp => (
                                                                    <Form.Item
                                                                        key={hp.name}
                                                                        label={
                                                                            <Tooltip
                                                                                title={hp.description || hp.name}
                                                                                placement="right"
                                                                            >
                                                                                <Space size={4}>
                                                                                    <Text code style={{ fontSize: 12 }}>{hp.name}</Text>
                                                                                    {hp.description && (
                                                                                        <Text type="secondary" style={{ fontSize: 11 }}>ⓘ</Text>
                                                                                    )}
                                                                                </Space>
                                                                            </Tooltip>
                                                                        }
                                                                        style={{ marginBottom: 20 }}
                                                                    >
                                                                        {renderControl(hp)}
                                                                    </Form.Item>
                                                                ))}
                                                            </div>
                                                        ),
                                                    }))}
                                                />
                                            );
                                        })()
                                    ) : (
                                        // Grid/Random/Bayesian Mode: checkboxes for select, min/max for numeric
                                        selectedAlgo?.hyperparameters?.map((hp: any) => {
                                            const isNumeric = hp.type === 'int' || hp.type === 'float';
                                            const isDiscreteGrid = tuningMethod === 'grid' && isNumeric;

                                            return (
                                                <Form.Item
                                                    key={hp.name}
                                                    label={
                                                        <Tooltip title={hp.description || hp.name} placement="right">
                                                            <Space size={4}>
                                                                <Text strong style={{ fontSize: 14 }}>{hp.name}</Text>
                                                                {hp.description && <Text type="secondary" style={{ fontSize: 11 }}>ⓘ</Text>}
                                                            </Space>
                                                        </Tooltip>
                                                    }
                                                    style={{ marginBottom: 20 }}
                                                >
                                                    {hp.type === 'select' ? (
                                                        // ── Categorical: multi-select checkboxes ─────────────────────
                                                        <div>
                                                            <Text type="secondary" style={{ fontSize: 12 }}>Check all values to include in the search space:</Text>
                                                            <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                                                                {hp.options?.map((opt: any) => {
                                                                    const isLR = selectedAlgo?.id === 'logistic_regression';
                                                                    const LR_SOLVER_PENALTIES: Record<string, Array<string | null>> = {
                                                                        'lbfgs': ['l2', null],
                                                                        'liblinear': ['l1', 'l2'],
                                                                        'newton-cg': ['l2', null],
                                                                        'sag': ['l2', null],
                                                                        'saga': ['l1', 'l2', 'elasticnet', null],
                                                                    };
                                                                    const normLR = (v: any): any => {
                                                                        if (typeof v === 'string') {
                                                                            const s = v.trim().toLowerCase();
                                                                            if (s === 'none') return null;
                                                                            if (s === 'true') return true;
                                                                            if (s === 'false') return false;
                                                                        }
                                                                        return v;
                                                                    };
                                                                    const toStringPenalty = (p: string | null) => (p === null ? 'None' : p);

                                                                    const currentSelected: any[] = Array.isArray(hyperparameters[hp.name])
                                                                        ? hyperparameters[hp.name]
                                                                        : [hp.default ?? opt.value];
                                                                    const isChecked = currentSelected.includes(opt.value);

                                                                    // LR grid/random/bayesian guard rails:
                                                                    // - penalty: always show all options; selecting penalties narrows solver options
                                                                    // - solver: only filter after user touched penalty
                                                                    // - dual=True only valid if solver=[liblinear] and penalty=[l2]
                                                                    // - l1_ratio only valid if solver includes saga AND penalty is ONLY elasticnet
                                                                    let disabled = false;

                                                                    if (isLR && (hp.name === 'solver' || hp.name === 'dual' || hp.name === 'l1_ratio')) {
                                                                        const selectedPenaltiesRaw = Array.isArray(hyperparameters['penalty'])
                                                                            ? hyperparameters['penalty']
                                                                            : (hyperparameters['penalty'] !== undefined ? [hyperparameters['penalty']] : ['l2']);
                                                                        const selectedPenalties = selectedPenaltiesRaw.map(normLR) as Array<string | null>;

                                                                        const selectedSolversRaw = Array.isArray(hyperparameters['solver'])
                                                                            ? hyperparameters['solver']
                                                                            : (hyperparameters['solver'] !== undefined ? [hyperparameters['solver']] : ['lbfgs']);
                                                                        const selectedSolvers = selectedSolversRaw.map((s: any) => String(normLR(s)));

                                                                        const penaltiesSet = new Set(selectedPenalties.map(toStringPenalty));
                                                                        const solversSet = new Set(selectedSolvers);

                                                                        if (hp.name === 'solver' && lrPenaltyTouched) {
                                                                            const allowedSolvers = new Set<string>();
                                                                            for (const p of selectedPenalties) {
                                                                                for (const s of Object.keys(LR_SOLVER_PENALTIES)) {
                                                                                    if ((LR_SOLVER_PENALTIES[s] ?? []).includes(p)) allowedSolvers.add(s);
                                                                                }
                                                                            }
                                                                            if (!allowedSolvers.has(String(opt.value))) disabled = true;
                                                                        }

                                                                        if (hp.name === 'dual' && String(opt.value).toLowerCase() === 'true') {
                                                                            const ok = solversSet.size === 1 && solversSet.has('liblinear') && penaltiesSet.size === 1 && penaltiesSet.has('l2');
                                                                            if (!ok) disabled = true;
                                                                        }

                                                                        if (hp.name === 'l1_ratio') {
                                                                            const ok = solversSet.has('saga') && penaltiesSet.size === 1 && penaltiesSet.has('elasticnet');
                                                                            if (!ok) disabled = true;
                                                                        }
                                                                    }

                                                                    return (
                                                                        <Checkbox
                                                                            key={opt.value}
                                                                            checked={isChecked}
                                                                            disabled={disabled}
                                                                            onChange={(e) => {
                                                                                const prev: any[] = Array.isArray(hyperparameters[hp.name])
                                                                                    ? hyperparameters[hp.name]
                                                                                    : [hp.default ?? opt.value];
                                                                                const next = e.target.checked
                                                                                    ? [...prev, opt.value]
                                                                                    : prev.filter((v: any) => v !== opt.value);

                                                                                // penalty primary behavior for LR: once user selects penalty, start filtering solver
                                                                                if (selectedAlgo?.id === 'logistic_regression' && hp.name === 'penalty') {
                                                                                    setLrPenaltyTouched(true);
                                                                                }

                                                                                setHyperparameters((prevHp: Record<string, any>) => {
                                                                                    const updated: Record<string, any> = { ...prevHp, [hp.name]: next.length ? next : [opt.value] };

                                                                                    if (selectedAlgo?.id !== 'logistic_regression') return updated;

                                                                                    const LR_SOLVER_PENALTIES: Record<string, Array<string | null>> = {
                                                                                        'lbfgs': ['l2', null],
                                                                                        'liblinear': ['l1', 'l2'],
                                                                                        'newton-cg': ['l2', null],
                                                                                        'sag': ['l2', null],
                                                                                        'saga': ['l1', 'l2', 'elasticnet', null],
                                                                                    };
                                                                                    const normLR = (v: any): any => {
                                                                                        if (typeof v === 'string') {
                                                                                            const s = v.trim().toLowerCase();
                                                                                            if (s === 'none') return null;
                                                                                            if (s === 'true') return true;
                                                                                            if (s === 'false') return false;
                                                                                        }
                                                                                        return v;
                                                                                    };
                                                                                    const toStringPenalty = (p: string | null) => (p === null ? 'None' : p);

                                                                                    const selPenRaw = Array.isArray(updated['penalty']) ? updated['penalty'] : (updated['penalty'] !== undefined ? [updated['penalty']] : ['l2']);
                                                                                    const selPen = selPenRaw.map(normLR) as Array<string | null>;
                                                                                    const penSet = new Set(selPen.map(toStringPenalty));

                                                                                    const selSolRaw = Array.isArray(updated['solver']) ? updated['solver'] : (updated['solver'] !== undefined ? [updated['solver']] : ['lbfgs']);
                                                                                    const selSol = selSolRaw.map((s: any) => String(normLR(s)));
                                                                                    const solSet = new Set(selSol);

                                                                                    // If penalty was touched, prune solver selections to only allowed for chosen penalties
                                                                                    if (lrPenaltyTouched || hp.name === 'penalty') {
                                                                                        const allowed = new Set<string>();
                                                                                        for (const p of selPen) {
                                                                                            for (const s of Object.keys(LR_SOLVER_PENALTIES)) {
                                                                                                if ((LR_SOLVER_PENALTIES[s] ?? []).includes(p)) allowed.add(s);
                                                                                            }
                                                                                        }
                                                                                        const pruned = selSol.filter(s => allowed.has(s));
                                                                                        if (pruned.length === 0) {
                                                                                            // pick a safe default compatible with first selected penalty
                                                                                            const firstPenalty = selPen[0] ?? 'l2';
                                                                                            const fallback = Object.keys(LR_SOLVER_PENALTIES).find(s => (LR_SOLVER_PENALTIES[s] ?? []).includes(firstPenalty)) ?? 'lbfgs';
                                                                                            updated['solver'] = [fallback];
                                                                                        } else {
                                                                                            updated['solver'] = pruned;
                                                                                        }
                                                                                    }

                                                                                    // dual=True only safe if solver=[liblinear] & penalty=[l2]
                                                                                    const dualRaw = Array.isArray(updated['dual']) ? updated['dual'] : (updated['dual'] !== undefined ? [updated['dual']] : []);
                                                                                    const dualTrueSelected = dualRaw.some((d: any) => String(d).toLowerCase() === 'true');
                                                                                    const dualOk = solSet.size === 1 && solSet.has('liblinear') && penSet.size === 1 && penSet.has('l2');
                                                                                    if (dualTrueSelected && !dualOk) {
                                                                                        updated['dual'] = ['False'];
                                                                                    }

                                                                                    // l1_ratio: only safe if penalty is ONLY elasticnet and solver includes saga.
                                                                                    // Otherwise remove it entirely to prevent sklearn from erroring in CV folds.
                                                                                    const l1Ok = solSet.has('saga') && penSet.size === 1 && penSet.has('elasticnet');
                                                                                    if (!l1Ok) {
                                                                                        delete updated['l1_ratio'];
                                                                                    }

                                                                                    return updated;
                                                                                });
                                                                            }}
                                                                        >
                                                                            <Text style={{ fontSize: 13 }}>{opt.label ?? opt.value}</Text>
                                                                        </Checkbox>
                                                                    );
                                                                })}
                                                            </div>
                                                        </div>
                                                    ) : isDiscreteGrid ? (
                                                        // ── Grid search: comma-separated explicit values ──────────────
                                                        <div>
                                                            <Text type="secondary" style={{ fontSize: 12 }}>Explicit Array Values (Comma-Separated)</Text>
                                                            <Input
                                                                placeholder={`e.g. ${hp.type === 'int' ? '100, 200, 300' : '0.1, 0.5, 1.0'}`}
                                                                defaultValue={hp.default}
                                                                onChange={(e) => {
                                                                    const valStr = e.target.value;
                                                                    const parsedList = valStr.split(',').map((s: string) => {
                                                                        const trimmed = s.trim();
                                                                        if (trimmed.toLowerCase() === 'true') return true;
                                                                        if (trimmed.toLowerCase() === 'false') return false;
                                                                        const num = hp.type === 'int' ? parseInt(trimmed, 10) : parseFloat(trimmed);
                                                                        return isNaN(num) ? trimmed : num;
                                                                    }).filter((v: any) => v !== '');
                                                                    if (parsedList.length > 0) {
                                                                        setHyperparameters({ ...hyperparameters, [hp.name]: parsedList });
                                                                    }
                                                                }}
                                                                style={{ marginTop: 4 }}
                                                            />
                                                        </div>
                                                    ) : (
                                                        // ── Numeric: min/max range → randint / uniform ───────────────
                                                        <>
                                                            <Row gutter={12}>
                                                                <Col span={12}>
                                                                    <Text type="secondary" style={{ fontSize: 12 }}>Min Value</Text>
                                                                    <Input
                                                                        type="number"
                                                                        placeholder={`Min (${hp.min})`}
                                                                        defaultValue={hp.min}
                                                                        step={hp.type === 'float' ? 0.01 : 1}
                                                                        onChange={(e) => setHyperparameters({
                                                                            ...hyperparameters,
                                                                            [hp.name]: { ...hyperparameters[hp.name], min: parseFloat(e.target.value) || hp.min }
                                                                        })}
                                                                        style={{ marginTop: 4 }}
                                                                    />
                                                                </Col>
                                                                <Col span={12}>
                                                                    <Text type="secondary" style={{ fontSize: 12 }}>Max Value</Text>
                                                                    <Input
                                                                        type="number"
                                                                        placeholder={`Max (${hp.max})`}
                                                                        defaultValue={hp.max}
                                                                        step={hp.type === 'float' ? 0.01 : 1}
                                                                        onChange={(e) => setHyperparameters({
                                                                            ...hyperparameters,
                                                                            [hp.name]: { ...hyperparameters[hp.name], max: parseFloat(e.target.value) || hp.max }
                                                                        })}
                                                                        style={{ marginTop: 4 }}
                                                                    />
                                                                </Col>
                                                            </Row>
                                                            {tuningMethod === 'grid' && (
                                                                <div style={{ marginTop: 8 }}>
                                                                    <Text type="secondary" style={{ fontSize: 12 }}>Step Size</Text>
                                                                    <Input
                                                                        type="number"
                                                                        placeholder="Step"
                                                                        defaultValue={hp.type === 'int' ? 1 : 0.1}
                                                                        step={hp.type === 'float' ? 0.01 : 1}
                                                                        onChange={(e) => setHyperparameters({
                                                                            ...hyperparameters,
                                                                            [hp.name]: { ...hyperparameters[hp.name], step: parseFloat(e.target.value) || (hp.type === 'int' ? 1 : 0.1) }
                                                                        })}
                                                                        style={{ marginTop: 4, width: '100%' }}
                                                                    />
                                                                </div>
                                                            )}
                                                        </>
                                                    )}
                                                </Form.Item>
                                            );
                                        })
                                    )}
                                </Card>
                            </Col>
                        </Row>
                    </div>
                );

            default:
                return null;
        }
    };

    return (
        <div className="fade-in">
            <div className="page-header">
                <div>
                    <Title level={2} style={{ margin: 0 }}>Training</Title>
                    <Text type="secondary">Configure and train anomaly detection models</Text>
                </div>
            </div>

            <Card>
                <Steps
                    current={currentStep}
                    onChange={(step) => setCurrentStep(step)}
                    className="site-navigation-steps"
                    style={{ marginTop: 24, cursor: 'pointer' }}
                    items={steps.map((step) => ({
                        title: step.title,
                        icon: step.icon,
                    }))}
                />
            </Card>

            {renderStepContent()}

            <div style={{ marginTop: 24, display: 'flex', justifyContent: 'space-between' }}>
                {currentStep > 0 && currentStep < 3 && (
                    <Button onClick={() => setCurrentStep(currentStep - 1)}>
                        Previous
                    </Button>
                )}
                {currentStep < 2 && (
                    <Button
                        type="primary"
                        onClick={() => setCurrentStep(currentStep + 1)}
                        disabled={
                            (currentStep === 0 && splittableDatasets.length === 0) || // Allow Next if any dataset exists (single or merged)
                            (currentStep === 1 && !selectedSplitJob)
                        }
                        style={{ marginLeft: 'auto' }}
                    >
                        Next
                    </Button>
                )}
                {/* Step 2 Special Next Button */}
                {currentStep === 2 && (
                    <Button
                        type="primary"
                        onClick={() => setCurrentStep(currentStep + 1)}
                        style={{ marginLeft: 'auto' }}
                        disabled={!featureSetId}
                    >
                        Proceed to Training
                    </Button>
                )}
                {currentStep === 3 && (
                    <Button onClick={() => setCurrentStep(0)} style={{ marginLeft: 'auto' }}>
                        Train New Model
                    </Button>
                )}
            </div>

            <Modal
                title={`Dataset Details: ${viewDataset?.name || ''}`}
                open={viewModalOpen}
                onCancel={() => {
                    setViewModalOpen(false);
                    setPreviewData(null);
                }}
                footer={[
                    <Button key="close" onClick={() => {
                        setViewModalOpen(false);
                        setPreviewData(null);
                    }}>
                        Close
                    </Button>,
                    <Button
                        key="download"
                        type="primary"
                        icon={<DownloadOutlined />}
                        onClick={handleDownload}
                    >
                        Download
                    </Button>,
                ]}
                width={800}
            >
                {viewDataset && (
                    <Tabs
                        defaultActiveKey="preview"
                        items={[
                            {
                                key: 'preview',
                                label: 'Data Preview',
                                children: (
                                    <>
                                        {previewLoading ? (
                                            <div style={{ textAlign: 'center', padding: '40px' }}>Loading preview...</div>
                                        ) : previewData ? (
                                            <Table
                                                dataSource={previewData.rows.map((row: any, idx: number) => ({ key: idx, ...row as object }))}
                                                columns={previewData.columns.map((col: string) => ({
                                                    title: col,
                                                    dataIndex: col,
                                                    key: col,
                                                    ellipsis: true,
                                                }))}
                                                pagination={false}
                                                scroll={{ x: true }}
                                                size="small"
                                            />
                                        ) : (
                                            <div>No preview data available</div>
                                        )}
                                        {previewData && (
                                            <Text type="secondary" style={{ display: 'block', marginTop: 16 }}>
                                                Showing {previewData.rows.length} of {previewData.total_rows} rows
                                            </Text>
                                        )}
                                    </>
                                ),
                            },
                            {
                                key: 'info',
                                label: 'File Info',
                                children: (
                                    <div>
                                        <div style={{ marginBottom: 16 }}>
                                            <Tag color="blue">{viewDataset.file_format.toUpperCase()}</Tag>
                                            <Tag color="cyan">{viewDataset.row_count} rows</Tag>
                                            <Tag color="purple">{viewDataset.column_count?.toLocaleString()} columns</Tag>
                                            <Tag>{(viewDataset.file_size_bytes ? viewDataset.file_size_bytes / 1024 : 0).toFixed(1)} KB</Tag>
                                        </div>
                                        <Card type="inner" title="Schema" size="small">
                                            <div style={{ maxHeight: 300, overflow: 'auto' }}>
                                                {viewDataset.schema?.columns.map(col => (
                                                    <div key={col.name} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid #f0f0f0' }}>
                                                        <Text strong>{col.name}</Text>
                                                        <Tag>{col.type}</Tag>
                                                    </div>
                                                ))}
                                            </div>
                                        </Card>
                                        {viewDataset.description && (
                                            <div style={{ marginTop: 16 }}>
                                                <Text strong>Description:</Text>
                                                <Paragraph>{viewDataset.description}</Paragraph>
                                            </div>
                                        )}
                                    </div>
                                ),
                            },
                        ]}
                    />
                )}
            </Modal >

            <Modal
                title={`Merge ${selectedMergeDatasets.length} Datasets`}
                open={mergeModalOpen}
                onCancel={() => setMergeModalOpen(false)}
                onOk={mergeForm.submit}
                confirmLoading={mergeDatasetMutation.isPending}
            >
                <Alert
                    message="Ensure Compatibility"
                    description="All selected datasets must have the exact same column names and types."
                    type="info"
                    showIcon
                    style={{ marginBottom: 24 }}
                />
                <Form
                    form={mergeForm}
                    layout="vertical"
                    onFinish={(values) => {
                        mergeDatasetMutation.mutate({
                            ids: selectedMergeDatasets as string[],
                            name: values.name,
                            description: values.description
                        });
                    }}
                >
                    <Form.Item
                        name="name"
                        label="Merged Dataset Name"
                        rules={[{ required: true, message: 'Please enter a name' }]}
                    >
                        <Input placeholder="e.g., merged_data_v1" />
                    </Form.Item>
                    <Form.Item
                        name="description"
                        label="Description"
                    >
                        <Input.TextArea placeholder="Optional description" />
                    </Form.Item>
                </Form>
            </Modal>
            {/* ── Hyperparameter Conflict Warnings Modal ─────────────────────── */}
            <Modal
                title={
                    <Space>
                        <span style={{ fontSize: 18 }}>⚠️</span>
                        <span style={{ fontWeight: 600 }}>Hyperparameter Conflicts Detected</span>
                    </Space>
                }
                open={warningModalOpen}
                onOk={() => setWarningModalOpen(false)}
                onCancel={() => setWarningModalOpen(false)}
                okText="I understand — fix before next run"
                cancelText="Close"
                width={680}
                styles={{
                    header: { borderBottom: '1px solid #f0f0f0', paddingBottom: 12 },
                    body: { padding: '20px 24px' },
                }}
            >
                <Alert
                    message={jobRejected ? "The training job was REJECTED due to the following invalid combinations." : "The training job was queued, but the following hyperparameter combinations may cause it to fail."}
                    description={jobRejected ? "Fix the issues below to start training." : "Fix the issues below and retry to ensure successful training."}
                    type={jobRejected ? "error" : "warning"}
                    showIcon
                    style={{ marginBottom: 16 }}
                />
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                    {validationWarnings.map((w, i) => {
                        const isError = w.startsWith('❌');
                        return (
                            <div
                                key={i}
                                style={{
                                    background: isError ? '#fff2f0' : '#fffbe6',
                                    border: `1px solid ${isError ? '#ffccc7' : '#ffe58f'}`,
                                    borderRadius: 8,
                                    padding: '10px 14px',
                                    fontSize: 13,
                                    lineHeight: 1.6,
                                    color: '#333',
                                    whiteSpace: 'pre-wrap',
                                }}
                            >
                                {w}
                            </div>
                        );
                    })}
                </div>
            </Modal>
        </div >
    );
}
