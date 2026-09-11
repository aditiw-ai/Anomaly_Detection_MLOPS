/**
 * Retraining Page
 * Manage automated model retraining.
 */
import { useState } from 'react';
import {
    Card, Table, Tag, Button, Typography, Row, Col, Space,
    Modal, Form, Input, InputNumber, Select, Switch, Progress, Popconfirm,
    Statistic, Steps, message
} from 'antd';
import {
    SyncOutlined, PlayCircleOutlined, CheckCircleOutlined,
    CloseCircleOutlined, RocketOutlined, PlusOutlined, DeleteOutlined
} from '@ant-design/icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../../api/axios';
const { Title, Text } = Typography;
const { Option } = Select;

interface RetrainJob {
    id: string;
    model_id: string;
    reason: string;
    status: string;
    current_step: string;
    progress: number;
    started_at: string;
    completed_at?: string;
    new_model_id?: string;
}

const retrainingService = {
    listJobs: async () => {
        const response = await api.get(`/retraining`);
        return response.data;
    },
    triggerRetraining: async (data: any) => {
        const response = await api.post(`/retraining/trigger`, data);
        return response.data;
    },
    runJob: async (jobId: string) => {
        const response = await api.post(`/retraining/${jobId}/run`);
        return response.data;
    },
    promoteModel: async (jobId: string) => {
        const response = await api.post(`/retraining/${jobId}/promote`);
        return response.data;
    },
    deleteJob: async (jobId: string) => {
        const response = await api.delete(`/retraining/${jobId}`);
        return response.data;
    },
    getReasons: async () => {
        const response = await api.get(`/retraining/reasons/available`);
        return response.data;
    },
};

export function Retraining() {
    const [createModal, setCreateModal] = useState(false);
    const [form] = Form.useForm();
    const queryClient = useQueryClient();

    // Fetch jobs
    const { data: jobsData, isLoading } = useQuery({
        queryKey: ['retrain-jobs'],
        queryFn: retrainingService.listJobs,
    });

    // Fetch reasons
    const { data: reasonsData } = useQuery({
        queryKey: ['retrain-reasons'],
        queryFn: retrainingService.getReasons,
    });

    // Trigger mutation
    const triggerMutation = useMutation({
        mutationFn: retrainingService.triggerRetraining,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['retrain-jobs'] });
            setCreateModal(false);
            form.resetFields();
            message.success('Retraining triggered');
        },
    });

    // Run mutation
    const runMutation = useMutation({
        mutationFn: retrainingService.runJob,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['retrain-jobs'] });
            message.success('Retraining pipeline started');
        },
    });

    // Promote mutation
    const promoteMutation = useMutation({
        mutationFn: retrainingService.promoteModel,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['retrain-jobs'] });
            message.success('Model promoted to production');
        },
    });

    const deleteMutation = useMutation({
        mutationFn: retrainingService.deleteJob,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['retrain-jobs'] });
            message.success('Retraining job deleted');
        },
    });

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'COMPLETED': return 'green';
            case 'FAILED': return 'red';
            case 'REJECTED': return 'orange';
            case 'TRAINING':
            case 'VALIDATION':
            case 'COMPARISON': return 'blue';
            default: return 'default';
        }
    };

    const getReasonColor = (reason: string) => {
        switch (reason) {
            case 'DRIFT_DETECTED': return 'orange';
            case 'PERFORMANCE_DEGRADATION': return 'red';
            case 'BIAS_DETECTED': return 'purple';
            case 'SCHEDULED': return 'blue';
            default: return 'default';
        }
    };
    const jobs = jobsData?.data || [];
    const reasons = reasonsData?.data || [];
    const reasonDescriptions: Record<string, string> = Object.fromEntries(
        reasons.map((r: any) => [r.reason, r.description])
    );

    const columns = [
        {
            title: 'Model',
            dataIndex: 'model_id',
            key: 'model_id',
            render: (id: string) => <Text code>{id.slice(0, 8)}...</Text>,
        },
        {
            title: 'Reason',
            dataIndex: 'reason',
            key: 'reason',
            render: (reason: string) => (
                <div>
                    <Tag color={getReasonColor(reason)}>{reason.replace(/_/g, ' ')}</Tag>
                    <div>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                            {reasonDescriptions[reason] || 'Retraining trigger reason'}
                        </Text>
                    </div>
                </div>
            ),
        },
        {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',
            render: (status: string) => <Tag color={getStatusColor(status)}>{status}</Tag>,
        },
        {
            title: 'Progress',
            key: 'progress',
            render: (_: any, record: RetrainJob) => (
                <div style={{ width: 120 }}>
                    <Progress
                        percent={record.progress * 100}
                        size="small"
                        status={record.status === 'FAILED' ? 'exception' : undefined}
                    />
                    <Text type="secondary" style={{ fontSize: 12 }}>{record.current_step}</Text>
                </div>
            ),
        },
        {
            title: 'Started',
            dataIndex: 'started_at',
            key: 'started_at',
            render: (date: string) => new Date(date).toLocaleString(),
        },
        {
            title: 'Actions',
            key: 'actions',
            render: (_: any, record: RetrainJob) => (
                <Space>
                    {record.status === 'PENDING' && (
                        <Button
                            size="small"
                            icon={<PlayCircleOutlined />}
                            onClick={() => runMutation.mutate(record.id)}
                            loading={runMutation.isPending}
                        >
                            Run
                        </Button>
                    )}
                    {record.status === 'COMPLETED' && record.new_model_id && (
                        <Button
                            size="small"
                            type="primary"
                            icon={<RocketOutlined />}
                            onClick={() => promoteMutation.mutate(record.id)}
                        >
                            Promote
                        </Button>
                    )}
                    {(record.status === 'FAILED' || record.status === 'REJECTED') && (
                        <Popconfirm
                            title="Delete retraining job?"
                            description="This will remove the job from the retraining list."
                            okText="Delete"
                            okButtonProps={{ danger: true }}
                            onConfirm={() => deleteMutation.mutate(record.id)}
                        >
                            <Button
                                size="small"
                                danger
                                icon={<DeleteOutlined />}
                                loading={deleteMutation.isPending}
                            />
                        </Popconfirm>
                    )}
                </Space>
            ),
        },
    ];

    const runningJobs = jobs.filter((j: RetrainJob) =>
        ['TRAINING', 'VALIDATION', 'COMPARISON', 'DATA_PREPARATION'].includes(j.status)
    ).length;

    return (
        <div className="fade-in">
            <div className="page-header">
                <div>
                    <Title level={2} style={{ margin: 0 }}>Model Retraining</Title>
                    <Text type="secondary">Automated retraining pipeline</Text>
                </div>
                <Button
                    type="primary"
                    icon={<PlusOutlined />}
                    onClick={() => setCreateModal(true)}
                >
                    Trigger Retraining
                </Button>
            </div>

            {/* Summary Cards */}
            <Row gutter={16} style={{ marginBottom: 24 }}>
                <Col span={6}>
                    <Card>
                        <Statistic
                            title="Total Jobs"
                            value={jobs.length}
                            prefix={<SyncOutlined />}
                        />
                    </Card>
                </Col>
                <Col span={6}>
                    <Card>
                        <Statistic
                            title="Running"
                            value={runningJobs}
                            valueStyle={{ color: '#1890ff' }}
                        />
                    </Card>
                </Col>
                <Col span={6}>
                    <Card>
                        <Statistic
                            title="Completed"
                            value={jobs.filter((j: RetrainJob) => j.status === 'COMPLETED').length}
                            valueStyle={{ color: '#52c41a' }}
                            prefix={<CheckCircleOutlined />}
                        />
                    </Card>
                </Col>
                <Col span={6}>
                    <Card>
                        <Statistic
                            title="Failed/Rejected"
                            value={jobs.filter((j: RetrainJob) =>
                                j.status === 'FAILED' || j.status === 'REJECTED'
                            ).length}
                            valueStyle={{ color: '#ff4d4f' }}
                            prefix={<CloseCircleOutlined />}
                        />
                    </Card>
                </Col>
            </Row>

            {/* Pipeline Steps */}
            <Card title="Retraining Pipeline Steps" style={{ marginBottom: 24 }}>
                <Steps
                    items={[
                        { title: 'Data Prep', description: 'Fetch & prepare data' },
                        { title: 'Training', description: 'Train new model' },
                        { title: 'Validation', description: 'Evaluate performance' },
                        { title: 'Comparison', description: 'Compare with current' },
                        { title: 'Decision', description: 'Promote or reject' },
                    ]}
                />
            </Card>

            {/* Jobs Table */}
            <Card>
                <Table
                    loading={isLoading}
                    dataSource={jobs}
                    columns={columns}
                    rowKey="id"
                    pagination={{ pageSize: 10 }}
                />
            </Card>

            {/* Trigger Retraining Modal */}
            <Modal
                title="Trigger Model Retraining"
                open={createModal}
                onCancel={() => setCreateModal(false)}
                onOk={() => form.submit()}
                confirmLoading={triggerMutation.isPending}
                width={600}
            >
                <Form
                    form={form}
                    layout="vertical"
                    onFinish={(values) => triggerMutation.mutate(values)}
                    initialValues={{
                        algorithm: 'xgboost',
                        data_window_days: 90,
                        hyperparameter_tuning: true,
                        fairness_constraint: true,
                        auto_promote: false,
                    }}
                >
                    <Form.Item
                        name="model_id"
                        label="Model ID"
                        rules={[{ required: true, message: 'Please enter model ID' }]}
                    >
                        <Input placeholder="Model to retrain" />
                    </Form.Item>

                    <Form.Item
                        name="reason"
                        label="Reason"
                        rules={[{ required: true }]}
                    >
                        <Select placeholder="Select reason">
                            {reasons.map((r: any) => (
                                <Option key={r.reason} value={r.reason}>
                                    {r.reason.replace(/_/g, ' ')} - {r.description}
                                </Option>
                            ))}
                        </Select>
                    </Form.Item>

                    <Row gutter={16}>
                        <Col span={12}>
                            <Form.Item name="algorithm" label="Algorithm">
                                <Select>
                                    <Option value="xgboost">XGBoost</Option>
                                    <Option value="lightgbm">LightGBM</Option>
                                    <Option value="random_forest">Random Forest</Option>
                                </Select>
                            </Form.Item>
                        </Col>
                        <Col span={12}>
                            <Form.Item name="data_window_days" label="Data Window (days)">
                                <InputNumber min={7} max={365} style={{ width: '100%' }} />
                            </Form.Item>
                        </Col>
                    </Row>

                    <Row gutter={16}>
                        <Col span={8}>
                            <Form.Item name="hyperparameter_tuning" label="HP Tuning" valuePropName="checked">
                                <Switch />
                            </Form.Item>
                        </Col>
                        <Col span={8}>
                            <Form.Item name="fairness_constraint" label="Fairness" valuePropName="checked">
                                <Switch />
                            </Form.Item>
                        </Col>
                        <Col span={8}>
                            <Form.Item name="auto_promote" label="Auto-Promote" valuePropName="checked">
                                <Switch />
                            </Form.Item>
                        </Col>
                    </Row>
                </Form>
            </Modal>
        </div>
    );
}
