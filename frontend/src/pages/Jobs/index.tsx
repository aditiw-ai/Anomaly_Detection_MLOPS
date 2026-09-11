/**
 * Jobs Page
 * Manage scheduled monitoring jobs.
 */
import { useState } from 'react';
import {
    Card, Table, Tag, Button, Typography, Row, Col, Space,
    Modal, Form, Select, Input, Switch, message, Statistic, Tabs, Progress, Tooltip, Divider, Popconfirm
} from 'antd';
import {
    ScheduleOutlined, PlayCircleOutlined,
    PlusOutlined, DeleteOutlined, RocketOutlined, EyeOutlined, RedoOutlined
} from '@ant-design/icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { jobService, ScheduledJob } from '@/services/jobService';
import { trainingService } from '@/services/trainingService';

const { Title, Text } = Typography;
const { Option } = Select;

export function Jobs() {
    const [createModal, setCreateModal] = useState(false);
    const [form] = Form.useForm();
    const queryClient = useQueryClient();

    // Fetch scheduled jobs
    const { data: jobsData, isLoading } = useQuery({
        queryKey: ['jobs'],
        queryFn: () => jobService.listJobs(),
    });

    // Fetch training jobs with auto-refresh for running jobs
    const { data: trainingJobs, isLoading: trainingJobsLoading } = useQuery({
        queryKey: ['trainingJobs'],
        queryFn: () => trainingService.listJobs(),
        refetchInterval: (query) => {
            const hasRunningJobs = query.state.data?.data?.some(
                (job: any) => job.status === 'RUNNING' || job.status === 'QUEUED'
            );
            return hasRunningJobs ? 5000 : false;
        }
    });

    // Fetch job types
    const { data: typesData } = useQuery({
        queryKey: ['jobTypes'],
        queryFn: () => jobService.getJobTypes(),
    });

    // Create job mutation
    const createMutation = useMutation({
        mutationFn: jobService.createJob,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['jobs'] });
            setCreateModal(false);
            form.resetFields();
            message.success('Job created');
        },
    });

    // Run job mutation
    const runMutation = useMutation({
        mutationFn: jobService.runJob,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['jobs'] });
            message.success('Job triggered');
        },
    });

    // Enable/disable mutation
    const toggleMutation = useMutation({
        mutationFn: ({ jobId, enabled }: { jobId: string; enabled: boolean }) =>
            enabled ? jobService.enableJob(jobId) : jobService.disableJob(jobId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['jobs'] });
        },
    });

    // Delete mutation
    const deleteMutation = useMutation({
        mutationFn: jobService.deleteJob,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['jobs'] });
            message.success('Job deleted');
        },
    });

    // Delete training job mutation
    const deleteTrainingJobMutation = useMutation({
        mutationFn: trainingService.deleteJob,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['trainingJobs'] });
            message.success('Training job deleted');
        },
    });

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'COMPLETED': return 'green';
            case 'RUNNING': return 'blue';
            case 'FAILED': return 'red';
            case 'PENDING': return 'default';
            default: return 'default';
        }
    };

    const getJobTypeLabel = (type: string) => {
        const labels: Record<string, string> = {
            DRIFT_CHECK: 'Drift Check',
            BIAS_CHECK: 'Bias Check',
            PERFORMANCE_CHECK: 'Performance Check',
            MODEL_RETRAIN: 'Model Retrain',
            DATA_CLEANUP: 'Data Cleanup',
        };
        return labels[type] || type;
    };

    const columns = [
        {
            title: 'Job Type',
            dataIndex: 'job_type',
            key: 'job_type',
            render: (type: string) => (
                <Tag color="blue">{getJobTypeLabel(type)}</Tag>
            ),
        },
        {
            title: 'Schedule',
            dataIndex: 'schedule',
            key: 'schedule',
            render: (schedule: string) => (
                <Text code>{schedule}</Text>
            ),
        },
        {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',
            render: (status: string) => (
                <Tag color={getStatusColor(status)}>{status}</Tag>
            ),
        },
        {
            title: 'Enabled',
            dataIndex: 'enabled',
            key: 'enabled',
            render: (enabled: boolean, record: ScheduledJob) => (
                <Switch
                    checked={enabled}
                    onChange={(checked) => toggleMutation.mutate({ jobId: record.id, enabled: checked })}
                    loading={toggleMutation.isPending}
                />
            ),
        },
        {
            title: 'Last Run',
            dataIndex: 'last_run',
            key: 'last_run',
            render: (date: string) => date ? new Date(date).toLocaleString() : '-',
        },
        {
            title: 'Next Run',
            dataIndex: 'next_run',
            key: 'next_run',
            render: (date: string) => new Date(date).toLocaleString(),
        },
        {
            title: 'Actions',
            key: 'actions',
            render: (_: any, record: ScheduledJob) => (
                <Space>
                    <Button
                        size="small"
                        icon={<PlayCircleOutlined />}
                        onClick={() => runMutation.mutate(record.id)}
                        loading={runMutation.isPending}
                    >
                        Run Now
                    </Button>
                    <Button
                        size="small"
                        danger
                        icon={<DeleteOutlined />}
                        onClick={() => {
                            Modal.confirm({
                                title: 'Delete Job',
                                content: 'Are you sure you want to delete this job?',
                                onOk: () => deleteMutation.mutate(record.id),
                            });
                        }}
                    />
                </Space>
            ),
        },
    ];

    const jobs = jobsData?.data || [];
    const activeCount = jobs.filter((j: ScheduledJob) => j.enabled).length;

    // Training jobs columns
    const trainingJobsColumns = [
        {
            title: 'Job Name',
            dataIndex: 'name',
            key: 'name',
            render: (text: string) => <Text strong>{text}</Text>
        },
        {
            title: 'Algorithm',
            dataIndex: 'algorithm',
            key: 'algorithm',
            width: 120,
            render: (algo: string) => (
                <Tag color="purple">{algo?.toUpperCase()}</Tag>
            )
        },
        {
            title: 'Hyperparameters',
            dataIndex: 'hyperparameters',
            key: 'hyperparameters',
            width: 150,
            render: (params: Record<string, any>) => (
                <Tooltip
                    title={
                        <pre style={{ margin: 0, fontSize: 11, maxHeight: 300, overflow: 'auto' }}>
                            {JSON.stringify(params, null, 2)}
                        </pre>
                    }
                >
                    <Tag style={{ cursor: 'pointer' }}>
                        {Object.keys(params || {}).length} parameters
                    </Tag>
                </Tooltip>
            )
        },
        {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',
            width: 120,
            render: (status: string) => {
                const colors: Record<string, string> = {
                    QUEUED: 'blue',
                    RUNNING: 'orange',
                    COMPLETED: 'green',
                    FAILED: 'red',
                    DATA_PREPARED: 'cyan'
                };
                return <Tag color={colors[status] || 'default'}>{status}</Tag>;
            }
        },
        {
            title: 'Progress',
            dataIndex: 'progress',
            key: 'progress',
            width: 150,
            render: (progress: number) => (
                <Progress
                    percent={Math.round((progress || 0) * 100)}
                    size="small"
                    status={progress === 1 ? 'success' : 'active'}
                />
            )
        },
        {
            title: 'Metrics',
            dataIndex: 'metrics',
            key: 'metrics',
            width: 120,
            render: (metrics: any) => {
                if (!metrics?.precision) return <Text type="secondary">-</Text>;
                return (
                    <Space direction="vertical" size={0}>
                        <Text style={{ fontSize: 11 }}>F1: {metrics.f1?.toFixed(3)}</Text>
                        <Text style={{ fontSize: 11 }}>AUC: {metrics.auc?.toFixed(3)}</Text>
                    </Space>
                );
            }
        },
        {
            title: 'Created',
            dataIndex: 'created_at',
            key: 'created',
            width: 150,
            render: (date: string) => new Date(date).toLocaleString()
        },
        {
            title: 'Actions',
            key: 'actions',
            width: 100,
            render: (_: any, record: any) => (
                <Space>
                    <Tooltip title="View Details">
                        <Button
                            icon={<EyeOutlined />}
                            size="small"
                            onClick={() => {
                                Modal.info({
                                    title: `Job: ${record.name}`,
                                    width: 700,
                                    content: (
                                        <div>
                                            <Divider />
                                            <Row gutter={[16, 16]}>
                                                <Col span={12}>
                                                    <Statistic title="Algorithm" value={record.algorithm?.toUpperCase()} />
                                                </Col>
                                                <Col span={12}>
                                                    <Statistic title="Status" value={record.status} />
                                                </Col>
                                                <Col span={12}>
                                                    <Statistic title="Progress" value={`${(record.progress * 100).toFixed(1)}%`} />
                                                </Col>
                                                <Col span={12}>
                                                    <Statistic title="Created" value={new Date(record.created_at).toLocaleString()} />
                                                </Col>
                                            </Row>

                                            <Divider orientation="left">Hyperparameters</Divider>
                                            <pre style={{
                                                background: '#f5f5f5',
                                                padding: 12,
                                                borderRadius: 4,
                                                maxHeight: 300,
                                                overflow: 'auto'
                                            }}>
                                                {JSON.stringify(record.hyperparameters, null, 2)}
                                            </pre>

                                            {record.metrics && Object.keys(record.metrics).length > 0 && (
                                                <>
                                                    <Divider orientation="left">Performance Metrics</Divider>
                                                    <Row gutter={[16, 16]}>
                                                        {record.metrics.precision && (
                                                            <Col span={6}>
                                                                <Statistic
                                                                    title="Precision"
                                                                    value={record.metrics.precision.toFixed(3)}
                                                                    precision={3}
                                                                />
                                                            </Col>
                                                        )}
                                                        {record.metrics.recall && (
                                                            <Col span={6}>
                                                                <Statistic
                                                                    title="Recall"
                                                                    value={record.metrics.recall.toFixed(3)}
                                                                    precision={3}
                                                                />
                                                            </Col>
                                                        )}
                                                        {record.metrics.f1 && (
                                                            <Col span={6}>
                                                                <Statistic
                                                                    title="F1 Score"
                                                                    value={record.metrics.f1.toFixed(3)}
                                                                    precision={3}
                                                                />
                                                            </Col>
                                                        )}
                                                        {record.metrics.auc && (
                                                            <Col span={6}>
                                                                <Statistic
                                                                    title="ROC AUC"
                                                                    value={record.metrics.auc.toFixed(3)}
                                                                    precision={3}
                                                                />
                                                            </Col>
                                                        )}
                                                    </Row>
                                                </>
                                            )}
                                        </div>
                                    )
                                });
                            }}
                        />
                    </Tooltip>
                    <Popconfirm
                        title="Delete this job?"
                        description="This action cannot be undone."
                        onConfirm={() => deleteTrainingJobMutation.mutate(record.id)}
                        okText="Yes"
                        cancelText="No"
                    >
                        <Tooltip title="Delete Job">
                            <Button
                                danger
                                icon={<DeleteOutlined />}
                                size="small"
                            />
                        </Tooltip>
                    </Popconfirm>
                </Space>
            )
        }
    ];

    return (
        <div className="fade-in">
            <div className="page-header">
                <div>
                    <Title level={2} style={{ margin: 0 }}>Scheduled Jobs</Title>
                    <Text type="secondary">Manage automated monitoring tasks</Text>
                </div>
                <Button
                    type="primary"
                    icon={<PlusOutlined />}
                    onClick={() => setCreateModal(true)}
                >
                    Create Job
                </Button>
            </div>

            {/* Tabs for Job Types */}
            <Tabs
                defaultActiveKey="training"
                items={[
                    {
                        key: 'training',
                        label: (
                            <span>
                                <RocketOutlined />
                                Training Jobs
                            </span>
                        ),
                        children: (
                            <Card
                                extra={
                                    <Button
                                        icon={<RedoOutlined />}
                                        onClick={() => queryClient.invalidateQueries({ queryKey: ['trainingJobs'] })}
                                    >
                                        Refresh
                                    </Button>
                                }
                            >
                                <Table
                                    dataSource={trainingJobs?.data?.filter((job: any) => !job.processing_only) || []}
                                    loading={trainingJobsLoading}
                                    rowKey="id"
                                    columns={trainingJobsColumns}
                                    pagination={{ pageSize: 10 }}
                                    size="small"
                                />
                            </Card>
                        )
                    },
                    {
                        key: 'scheduled',
                        label: (
                            <span>
                                <ScheduleOutlined />
                                Scheduled Jobs
                            </span>
                        ),
                        children: (
                            <>
                                {/* Summary Cards */}
                                <Row gutter={16} style={{ marginBottom: 24 }}>
                                    <Col span={6}>
                                        <Card>
                                            <Statistic
                                                title="Total Jobs"
                                                value={jobs.length}
                                                prefix={<ScheduleOutlined />}
                                            />
                                        </Card>
                                    </Col>
                                    <Col span={6}>
                                        <Card>
                                            <Statistic
                                                title="Active"
                                                value={activeCount}
                                                valueStyle={{ color: '#52c41a' }}
                                            />
                                        </Card>
                                    </Col>
                                    <Col span={6}>
                                        <Card>
                                            <Statistic
                                                title="Disabled"
                                                value={jobs.length - activeCount}
                                                valueStyle={{ color: '#8c8c8c' }}
                                            />
                                        </Card>
                                    </Col>
                                    <Col span={6}>
                                        <Card>
                                            <Statistic
                                                title="Job Types"
                                                value={typesData?.data?.length || 5}
                                            />
                                        </Card>
                                    </Col>
                                </Row>

                                {/* Jobs Table */}
                                <Card>
                                    <Table
                                        loading={isLoading}
                                        dataSource={jobs}
                                        columns={columns}
                                        rowKey="id"
                                        pagination={{
                                            pageSize: 10,
                                            showTotal: (total) => `Total ${total} jobs`,
                                        }}
                                    />
                                </Card>
                            </>
                        )
                    }
                ]}
            />

            {/* Create Job Modal */}
            <Modal
                title="Create Scheduled Job"
                open={createModal}
                onCancel={() => {
                    setCreateModal(false);
                    form.resetFields();
                }}
                onOk={() => form.submit()}
                confirmLoading={createMutation.isPending}
            >
                <Form
                    form={form}
                    layout="vertical"
                    onFinish={(values) => createMutation.mutate(values)}
                >
                    <Form.Item
                        name="job_type"
                        label="Job Type"
                        rules={[{ required: true, message: 'Please select a job type' }]}
                    >
                        <Select placeholder="Select job type">
                            {typesData?.data?.map((t: any) => (
                                <Option key={t.type} value={t.type}>
                                    {getJobTypeLabel(t.type)} - {t.description}
                                </Option>
                            ))}
                        </Select>
                    </Form.Item>

                    <Form.Item
                        name="schedule"
                        label="Schedule (Cron)"
                        tooltip="Cron expression: minute hour day month weekday"
                    >
                        <Input placeholder="0 * * * * (every hour)" />
                    </Form.Item>

                    <Form.Item
                        name="model_id"
                        label="Model ID (Optional)"
                    >
                        <Input placeholder="Leave blank for all models" />
                    </Form.Item>
                </Form>
            </Modal>
        </div>
    );
}
