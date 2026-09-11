/**
 * A/B Testing Page
 * Manage champion-challenger model tests.
 */
import { useEffect, useState } from 'react';
import {
    Card, Table, Tag, Button, Typography, Row, Col, Space,
    Modal, Form, Input, InputNumber, Select, Progress, Statistic, Spin
} from 'antd';
import {
    ExperimentOutlined, PlayCircleOutlined, StopOutlined,
    TrophyOutlined, CheckCircleOutlined, CloseCircleOutlined,
    PlusOutlined, LineChartOutlined, DeleteOutlined
} from '@ant-design/icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '../../api/axios';
import { modelService } from '../../services/modelService';
import { datasetService } from '../../services/datasetService';
import { message } from 'antd';

const { Title, Text } = Typography;
const { Option } = Select;

interface ABTest {
    id: string;
    name: string;
    champion_model_id: string;
    challenger_model_id: string;
    status: string;
    result: string;
    champion_samples: number;
    challenger_samples: number;
    champion_metrics?: any;
    challenger_metrics?: any;
    min_samples?: number;
    challenger_traffic_percent?: number;
    created_at: string;
}

interface SimulationProgress {
    status: string;
    phase: string;
    processed: number;
    total: number;
    percent: number;
    labelled_samples: number;
    message?: string;
    updated_at?: string;
}

type SimulateVars = {
    testId: string;
    dataset_id: string;
    rows: number;
    reset_existing: boolean;
};

const abTestingService = {
    listTests: async () => {
        const response = await api.get('/ab-tests');
        return response.data;
    },
    createTest: async (data: any) => {
        const response = await api.post(`/ab-tests`, data);
        return response.data;
    },

    startTest: async (testId: string) => {
        const response = await api.post(`/ab-tests/${testId}/start`);
        return response.data;
    },
    evaluateTest: async (id: string) => {
        const response = await api.post(`/ab-tests/${id}/evaluate`, {}, { params: { _ts: Date.now() } });
        return response.data;
    },
    simulationProgress: async (id: string) => {
        const response = await api.get(`/ab-tests/${id}/simulation-progress`, { params: { _ts: Date.now() } });
        return response.data;
    },
    abortTest: async (id: string, reason: string = "") => {
        const response = await api.post(`/ab-tests/${id}/abort`, {}, { params: { reason } });
        return response.data;
    },
    concludeTest: async (id: string, result: string, promote: boolean) => {
        const response = await api.post(`/ab-tests/${id}/conclude`, { result, promote_challenger: promote });
        return response.data;
    },
    simulateTest: async (testId: string, dataset_id: string, rows: number, reset_existing: boolean = true) => {
        // Simulation can take longer than the default API timeout on larger datasets.
        const response = await api.post(
            `/ab-tests/${testId}/simulate`,
            { dataset_id, rows, reset_existing },
            { timeout: 0 }
        );
        return response.data;
    },
    deleteTest: async (id: string) => {
        const response = await api.delete(`/ab-tests/${id}`);
        return response.data;
    }
};

export function ABTesting() {
    const queryClient = useQueryClient();
    const [newTestModal, setNewTestModal] = useState(false);
    const [evaluateModal, setEvaluateModal] = useState<string | null>(null);
    const [evaluationResult, setEvaluationResult] = useState<any>(null);
    const [evaluationLoading, setEvaluationLoading] = useState(false);
    const [simulateModal, setSimulateModal] = useState<string | null>(null);
    const [simulationInFlightTestId, setSimulationInFlightTestId] = useState<string | null>(null);
    const [simulationProgress, setSimulationProgress] = useState<SimulationProgress | null>(null);
    const [simulationStartedMessageShown, setSimulationStartedMessageShown] = useState(false);
    const [evaluationUpdatedAt, setEvaluationUpdatedAt] = useState<string | null>(null);
    const [form] = Form.useForm();
    const [simulateForm] = Form.useForm();

    const mergeSimulationProgress = (
        previous: SimulationProgress | null,
        incoming: SimulationProgress | null
    ): SimulationProgress | null => {
        if (!incoming) return previous;
        if (!previous) return incoming;

        const prevStatus = String(previous.status || '').toUpperCase();
        const nextStatus = String(incoming.status || '').toUpperCase();
        const prevProcessed = Number(previous.processed || 0);
        const nextProcessed = Number(incoming.processed || 0);
        const prevPercent = Number(previous.percent || 0);
        const nextPercent = Number(incoming.percent || 0);
        const nextLooksLikeFallbackIdle =
            nextStatus === 'IDLE' &&
            nextProcessed === 0 &&
            nextPercent === 0 &&
            prevProcessed > 0;

        // Ignore transient fallback payloads observed in live polling.
        if (nextLooksLikeFallbackIdle) {
            return previous;
        }

        const nextIsTerminal = nextStatus === 'COMPLETED' || nextStatus === 'FAILED';
        if (!nextIsTerminal && nextProcessed < prevProcessed) {
            return {
                ...incoming,
                processed: prevProcessed,
                total: Math.max(Number(previous.total || 0), Number(incoming.total || 0)),
                percent: Math.max(prevPercent, nextPercent),
                labelled_samples: Math.max(
                    Number(previous.labelled_samples || 0),
                    Number(incoming.labelled_samples || 0)
                ),
            };
        }

        return {
            ...incoming,
            total: Math.max(Number(previous.total || 0), Number(incoming.total || 0)),
            percent: nextIsTerminal ? nextPercent : Math.max(prevPercent, nextPercent),
            labelled_samples: Math.max(
                Number(previous.labelled_samples || 0),
                Number(incoming.labelled_samples || 0)
            ),
            // Keep an in-progress status if the incoming payload is stale/non-terminal.
            status:
                prevStatus === 'RUNNING' && nextStatus === 'IDLE' && !nextIsTerminal
                    ? 'RUNNING'
                    : incoming.status,
        };
    };

    const normalizeCol = (value: string) =>
        String(value || '')
            .toLowerCase()
            .replace(/[\s-]+/g, '_')
            .replace(/[^a-z0-9_]/g, '')
            .replace(/_+/g, '_')
            .replace(/^_+|_+$/g, '');

    const hasLikelyLabelColumn = (dataset: any) => {
        const cols: string[] = (dataset?.schema?.columns || []).map((c: any) => c?.name || '');
        const tokens = ['is_fraud', 'label', 'target', 'class', 'ground_truth', 'actual_label', 'fraud'];
        return cols.some((col) => {
            const n = normalizeCol(col);
            return tokens.some((t) => n === t || n.includes(t));
        });
    };

    const { data: tests, isLoading: testsLoading } = useQuery({
        queryKey: ['ab-tests'],
        queryFn: abTestingService.listTests
    });

    const { data: models } = useQuery({
        queryKey: ['models'],
        queryFn: () => modelService.listModels()
    });

    const { data: datasets } = useQuery({
        queryKey: ['datasets'],
        queryFn: () => datasetService.list()
    });

    const createMutation = useMutation({
        mutationFn: abTestingService.createTest,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['ab-tests'] });
            setNewTestModal(false);
            form.resetFields();
            message.success('A/B test created');
        },
    });

    const startMutation = useMutation({
        mutationFn: abTestingService.startTest,
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['ab-tests'] });
            message.success('A/B test started');
        },
        onError: (err: any) => {
            message.error(err.response?.data?.detail || 'Failed to start A/B test');
        },
    });

    const concludeMutation = useMutation({
        mutationFn: ({ testId, result, promote }: { testId: string; result: string; promote: boolean }) =>
            abTestingService.concludeTest(testId, result, promote),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['ab-tests'] });
            setEvaluateModal(null);
            setEvaluationResult(null);
        },
    });

    // Simulation mutation
    const simulateMutation = useMutation<any, any, SimulateVars>({
        mutationFn: ({ testId, dataset_id, rows, reset_existing }: SimulateVars) =>
            abTestingService.simulateTest(testId, dataset_id, rows, reset_existing),
        onMutate: (vars: SimulateVars) => {
            setSimulationInFlightTestId(vars.testId);
            setSimulationStartedMessageShown(false);
            setSimulationProgress({
                status: 'RUNNING',
                phase: 'INITIALIZING',
                processed: 0,
                total: vars.rows || 0,
                percent: 0,
                labelled_samples: 0,
            });
            setSimulateModal(null);
        },
        onSuccess: async (res: any) => {
            if (!simulationStartedMessageShown) {
                message.success(res?.message || 'Simulation started in background');
                setSimulationStartedMessageShown(true);
            }
        },
        onError: (err: any) => {
            message.error(err.response?.data?.detail || 'Simulation failed');
            setSimulationInFlightTestId(null);
            setSimulationProgress((prev) => ({
                ...(prev || { status: 'FAILED', phase: 'FAILED', processed: 0, total: 0, percent: 0, labelled_samples: 0 }),
                status: 'FAILED',
                phase: 'FAILED',
                message: err.response?.data?.detail || 'Simulation failed',
            }));
        }
    });

    const deleteMutation = useMutation({
        mutationFn: (testId: string) => abTestingService.deleteTest(testId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['ab-tests'] });
            message.success('A/B test deleted');
        },
        onError: (err: any) => {
            message.error(err.response?.data?.detail || 'Failed to delete A/B test');
        }
    });

    // Abort mutation
    const abortMutation = useMutation({
        mutationFn: (testId: string) => abTestingService.abortTest(testId),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['ab-tests'] });
            message.success('A/B test aborted');
        },
    });

    const fetchEvaluation = async (testId: string, silent = false) => {
        if (!silent) setEvaluationLoading(true);
        try {
            const res = await abTestingService.evaluateTest(testId);
            setEvaluationResult(res.data);
            setEvaluationUpdatedAt(new Date().toLocaleTimeString());
        } catch (error: any) {
            message.error(error.response?.data?.detail || 'Evaluation failed');
        } finally {
            if (!silent) setEvaluationLoading(false);
        }
    };

    useEffect(() => {
        if (!evaluateModal) return;
        if (!evaluationResult || evaluationResult.ready_for_decision) return;

        const t = setInterval(() => {
            fetchEvaluation(evaluateModal, true);
        }, 5000);

        return () => clearInterval(t);
    }, [evaluateModal, evaluationResult]);

    useEffect(() => {
        if (!simulationInFlightTestId) return;

        let stopped = false;
        let consecutive404s = 0;
        const tick = async () => {
            try {
                const res = await abTestingService.simulationProgress(simulationInFlightTestId);
                if (stopped) return;
                consecutive404s = 0;
                setSimulationProgress((prev) => mergeSimulationProgress(prev, res?.data || null));

                // Auto-stop polling when simulation finishes
                const status = String(res?.data?.status || '').toUpperCase();
                if (status === 'COMPLETED' || status === 'FAILED') {
                    stopped = true;
                    setSimulationInFlightTestId(null);
                    if (status === 'COMPLETED') {
                        queryClient.invalidateQueries({ queryKey: ['ab-tests'] });
                    }
                }
            } catch (err: any) {
                if (err?.response?.status === 404) {
                    consecutive404s++;
                    // Stop polling after 3 consecutive 404s — test not found in service
                    if (consecutive404s >= 3) {
                        stopped = true;
                        setSimulationInFlightTestId(null);
                        setSimulationProgress(null);
                    }
                }
                // Other errors: keep spinner alive; mutation handler will surface final failure.
            }
        };

        tick();
        const t = setInterval(tick, 2000);
        return () => {
            stopped = true;
            clearInterval(t);
        };
    }, [simulationInFlightTestId]);

    useEffect(() => {
        if (!simulationInFlightTestId || !simulationProgress) return;

        const status = String(simulationProgress.status || '').toUpperCase();
        if (status === 'RUNNING' || status === 'IDLE') return;

        const finalize = async () => {
            if (status === 'FAILED') {
                message.error(simulationProgress.message || 'Simulation failed');
            } else {
                const labelled = Number(simulationProgress.labelled_samples || 0);
                if (labelled > 0) {
                    message.success('Simulation completed successfully');
                } else {
                    message.warning(
                        "Simulation completed, but no usable label column was detected for comparison. " +
                        "Use a dataset with binary target labels (0/1, true/false, yes/no)."
                    );
                }
            }

            queryClient.invalidateQueries({ queryKey: ['ab-tests'] });
            const completedTestId = simulationInFlightTestId;
            setSimulationInFlightTestId(null);
            setSimulationProgress(null);
            setSimulationStartedMessageShown(false);
            simulateForm.resetFields();
            setEvaluateModal(completedTestId);
            await fetchEvaluation(completedTestId);
        };

        void finalize();
    }, [simulationProgress, simulationInFlightTestId]);

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'RUNNING': return 'blue';
            case 'COMPLETED': return 'green';
            case 'ABORTED': return 'red';
            default: return 'default';
        }
    };

    const getResultIcon = (result: string) => {
        switch (result) {
            case 'CHALLENGER_WINS': return <TrophyOutlined style={{ color: '#52c41a' }} />;
            case 'CHAMPION_WINS': return <CheckCircleOutlined style={{ color: '#1890ff' }} />;
            case 'NO_SIGNIFICANT_DIFFERENCE': return <CloseCircleOutlined style={{ color: '#8c8c8c' }} />;
            default: return null;
        }
    };

    const columns = [
        { title: 'Name', dataIndex: 'name', key: 'name' },
        {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',
            render: (status: string) => <Tag color={getStatusColor(status)}>{status}</Tag>,
        },
        {
            title: 'Result',
            dataIndex: 'result',
            key: 'result',
            render: (result: string, record: ABTest) => {
                const totalSamples = record.champion_samples + record.challenger_samples;
                const labelledSamples =
                    Number(record.champion_metrics?.samples || 0) +
                    Number(record.challenger_metrics?.samples || 0);
                const isReady =
                    record.status === 'RUNNING' &&
                    result === 'PENDING' &&
                    totalSamples >= (record.min_samples || 1000) &&
                    labelledSamples > 0;

                return (
                    <Space>
                        {getResultIcon(result)}
                        <Text>{result.replace(/_/g, ' ')}</Text>
                        {isReady && <Tag color="gold">READY</Tag>}
                    </Space>
                );
            },
        },
        {
            title: 'Split',
            key: 'split',
            render: (record: ABTest) => {
                const total = record.champion_samples + record.challenger_samples;
                const actual = total > 0 ? (record.challenger_samples / total) * 100 : 0;
                const target = record.challenger_traffic_percent ?? 10;
                return (
                    <Text type="secondary">
                        {actual.toFixed(1)}% / {target.toFixed(1)}%
                    </Text>
                );
            }
        },
        {
            title: 'Champ F1',
            key: 'champ_f1',
            render: (record: ABTest) => {
                const f1 = record.champion_metrics?.f1;
                return f1 !== undefined ? <Text strong style={{ color: '#1890ff' }}>{f1.toFixed(3)}</Text> : <Text type="secondary">-</Text>;
            }
        },
        {
            title: 'Chal F1',
            key: 'chal_f1',
            render: (record: ABTest) => {
                const f1 = record.challenger_metrics?.f1;
                return f1 !== undefined ? <Text strong style={{ color: '#52c41a' }}>{f1.toFixed(3)}</Text> : <Text type="secondary">-</Text>;
            }
        },
        {
            title: 'Champion Samples',
            dataIndex: 'champion_samples',
            key: 'champion_samples',
            render: (v: number) => v.toLocaleString(),
        },
        {
            title: 'Challenger Samples',
            dataIndex: 'challenger_samples',
            key: 'challenger_samples',
            render: (v: number) => v.toLocaleString(),
        },
        {
            title: 'Actions',
            key: 'actions',
            render: (_: any, record: ABTest) => (
                <Space>
                    {record.status === 'DRAFT' && (
                        <Button
                            type="primary"
                            icon={<PlayCircleOutlined />}
                            onClick={() => startMutation.mutate(record.id)}
                        >
                            Start
                        </Button>
                    )}
                    {record.status === 'RUNNING' && (
                        <>
                            <Button
                                icon={<PlayCircleOutlined />}
                                onClick={() => setSimulateModal(record.id)}
                            >
                                Simulate
                            </Button>
                            <Button
                                icon={<LineChartOutlined />}
                                type="primary"
                                ghost
                                onClick={async () => {
                                    setEvaluateModal(record.id);
                                    await fetchEvaluation(record.id);
                                }}
                            >
                                Evaluate
                            </Button>
                        </>
                    )}
                    <Button
                        icon={<StopOutlined />}
                        danger
                        loading={abortMutation.isPending}
                        onClick={() => {
                            Modal.confirm({
                                title: 'Abort A/B Test',
                                content: 'Are you sure you want to abort this test? This action cannot be undone.',
                                okText: 'Yes, Abort',
                                okType: 'danger',
                                cancelText: 'Cancel',
                                onOk: () => abortMutation.mutate(record.id)
                            });
                        }}
                    >
                        Abort
                    </Button>
                    <Button
                        icon={<DeleteOutlined />}
                        danger
                        loading={deleteMutation.isPending}
                        onClick={() => {
                            Modal.confirm({
                                title: 'Delete A/B Test',
                                content: 'This will permanently remove this test and its counters. Continue?',
                                okText: 'Delete',
                                okType: 'danger',
                                cancelText: 'Cancel',
                                onOk: () => deleteMutation.mutate(record.id)
                            });
                        }}
                    >

                    </Button>
                </Space>
            ),
        },
    ];

    const stats = [
        { title: 'Total Tests', value: tests?.meta?.total || 0, icon: <ExperimentOutlined /> },
        { title: 'Running', value: tests?.data?.filter((t: any) => t.status === 'RUNNING')?.length || 0, icon: <Progress type="circle" percent={0} size={20} />, color: '#1890ff' },
        { title: 'Challenger Wins', value: tests?.data?.filter((t: any) => t.result === 'CHALLENGER_WINS')?.length || 0, color: '#52c41a' },
        { title: 'Champion Wins', value: tests?.data?.filter((t: any) => t.result === 'CHAMPION_WINS')?.length || 0, color: '#1890ff' },
    ];
    const simulationEligibleDatasets = (datasets?.data || []).filter((d: any) => hasLikelyLabelColumn(d));

    return (
        <div style={{ padding: 24 }}>
            <Row gutter={[16, 16]} justify="space-between" align="middle" style={{ marginBottom: 24 }}>
                <Col>
                    <Title level={2}>A/B Testing</Title>
                    <Text type="secondary">Champion-challenger model comparison</Text>
                </Col>
                <Col>
                    <Button type="primary" icon={<PlusOutlined />} onClick={() => setNewTestModal(true)}>
                        New Test
                    </Button>
                </Col>
            </Row>

            <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                {stats.map((stat, i) => (
                    <Col xs={24} sm={12} md={6} key={i}>
                        <Card bordered={false}>
                            <Statistic
                                title={stat.title}
                                value={stat.value}
                                prefix={stat.icon}
                                valueStyle={{ color: stat.color }}
                            />
                        </Card>
                    </Col>
                ))}
            </Row>

            <Card bordered={false}>
                <Table
                    columns={columns}
                    dataSource={tests?.data || []}
                    rowKey="id"
                    loading={testsLoading}
                    pagination={{ pageSize: 10 }}
                />
            </Card>

            {/* New Test Modal */}
            <Modal
                title="Create A/B Test"
                open={newTestModal}
                onCancel={() => setNewTestModal(false)}
                onOk={() => form.submit()}
                confirmLoading={createMutation.isPending}
                width={800}
            >
                <Form form={form} layout="vertical" onFinish={(values) => createMutation.mutate(values)}>
                    <Form.Item
                        name="name"
                        label="Test Name"
                        rules={[{ required: true, message: 'Please enter a test name' }]}
                    >
                        <Input placeholder="e.g., XGBoost v2 vs Production" />
                    </Form.Item>

                    <Row gutter={16}>
                        <Col span={12}>
                            <Form.Item
                                name="champion_model_id"
                                label="Champion Model"
                                rules={[{ required: true, message: 'Please select a champion model' }]}
                            >
                                <Select placeholder="Select champion model" showSearch optionFilterProp="children">
                                    {models?.data?.filter((m: any) => m.status !== 'DELETED').map((m: any) => (
                                        <Option key={m.id} value={m.id}>
                                            {`${m.name} v${m.version} (${m.algorithm}) - ${m.created_at ? new Date(m.created_at).toLocaleDateString() : 'N/A'}`}
                                            {m.status === 'PRODUCTION' && <Tag color="gold" style={{ marginLeft: 8 }}>PROD</Tag>}
                                        </Option>
                                    ))}
                                </Select>
                            </Form.Item>
                        </Col>
                        <Col span={12}>
                            <Form.Item
                                name="challenger_model_id"
                                label="Challenger Model"
                                rules={[{ required: true, message: 'Please select a challenger model' }]}
                            >
                                <Select placeholder="Select challenger model" showSearch optionFilterProp="children">
                                    {models?.data?.filter((m: any) => m.status !== 'DELETED').map((m: any) => (
                                        <Option key={m.id} value={m.id}>
                                            {`${m.name} v${m.version} (${m.algorithm}) - ${m.created_at ? new Date(m.created_at).toLocaleDateString() : 'N/A'}`}
                                            {m.status === 'STAGING' && <Tag color="blue" style={{ marginLeft: 8 }}>STAGING</Tag>}
                                        </Option>
                                    ))}
                                </Select>
                            </Form.Item>
                        </Col>
                    </Row>

                    <Row gutter={16}>
                        <Col span={12}>
                            <Form.Item
                                name="challenger_traffic_percent"
                                label="Challenger Traffic %"
                                initialValue={10}
                            >
                                <InputNumber min={1} max={50} style={{ width: '100%' }} />
                            </Form.Item>
                        </Col>
                        <Col span={12}>
                            <Form.Item
                                name="min_samples"
                                label="Min Samples"
                                initialValue={1000}
                            >
                                <InputNumber min={100} style={{ width: '100%' }} />
                            </Form.Item>
                        </Col>
                    </Row>

                    <Form.Item
                        name="primary_metric"
                        label="Primary Metric"
                        initialValue="f1"
                    >
                        <Select>
                            <Option value="f1">F1 Score</Option>
                            <Option value="precision">Precision</Option>
                            <Option value="recall">Recall</Option>
                            <Option value="auc">AUC</Option>
                        </Select>
                    </Form.Item>
                </Form>
            </Modal>

            {/* Evaluation Modal */}
            <Modal
                title="A/B Test Evaluation"
                open={!!evaluateModal}
                onCancel={() => {
                    setEvaluateModal(null);
                    setEvaluationResult(null);
                }}
                footer={evaluationResult?.ready_for_decision ? [
                    <Button key="keep" onClick={() => evaluateModal && concludeMutation.mutate({
                        testId: evaluateModal, result: 'CHAMPION_WINS', promote: false
                    })}>
                        Keep Champion
                    </Button>,
                    <Button key="promote" type="primary" onClick={() => evaluateModal && concludeMutation.mutate({
                        testId: evaluateModal, result: 'CHALLENGER_WINS', promote: true
                    })}>
                        Promote Challenger
                    </Button>,
                ] : null}
                width={600}
            >
                {evaluationResult && (
                    <div style={{ textAlign: 'center' }}>
                        <Statistic
                            title="Decision Readiness"
                            value={evaluationResult.ready_for_decision ? 'READY' : 'COLLECTING DATA'}
                            valueStyle={{ color: evaluationResult.ready_for_decision ? '#52c41a' : '#faad14' }}
                        />
                        {evaluationResult.recommendation && (
                            <Tag
                                color={
                                    evaluationResult.recommendation === 'PROMOTE_CHALLENGER'
                                        ? 'green'
                                        : evaluationResult.recommendation === 'KEEP_CHAMPION'
                                            ? 'blue'
                                            : 'orange'
                                }
                                style={{ marginTop: 12 }}
                            >
                                Recommendation: {String(evaluationResult.recommendation).replace(/_/g, ' ')}
                            </Tag>
                        )}
                        <Row gutter={16} style={{ marginTop: 24 }}>
                            <Col span={12}>
                                <Card size="small" title="Champion F1">
                                    <Title level={3} style={{ color: '#1890ff' }}>
                                        {evaluationResult.champion_metrics?.f1?.toFixed(4) || 'N/A'}
                                    </Title>
                                </Card>
                            </Col>
                            <Col span={12}>
                                <Card size="small" title="Challenger F1">
                                    <Title level={3} style={{ color: '#52c41a' }}>
                                        {evaluationResult.challenger_metrics?.f1?.toFixed(4) || 'N/A'}
                                    </Title>
                                </Card>
                            </Col>
                        </Row>
                        <Row gutter={16} style={{ marginTop: 16 }}>
                            <Col span={12}>
                                <Card size="small" title="Traffic Samples">
                                    <Text strong>
                                        {Number(evaluationResult.samples_collected || 0).toLocaleString()} / {Number(evaluationResult.samples_needed || 0).toLocaleString()}
                                    </Text>
                                </Card>
                            </Col>
                            <Col span={12}>
                                <Card size="small" title="Labelled Samples">
                                    <Text strong>{Number(evaluationResult.labelled_samples_collected || 0).toLocaleString()}</Text>
                                </Card>
                            </Col>
                        </Row>
                        {evaluationResult.analysis && (
                            <Row gutter={16} style={{ marginTop: 16 }}>
                                <Col span={8}>
                                    <Card size="small" title="Primary Metric">
                                        <Text strong>{String(evaluationResult.analysis.primary_metric || '').toUpperCase()}</Text>
                                    </Card>
                                </Col>
                                <Col span={8}>
                                    <Card size="small" title="Difference">
                                        <Text strong>{Number(evaluationResult.analysis.difference || 0).toFixed(4)}</Text>
                                    </Card>
                                </Col>
                                <Col span={8}>
                                    <Card size="small" title="Significance">
                                        <Tag color={evaluationResult.analysis.is_significant ? 'green' : 'orange'}>
                                            {evaluationResult.analysis.is_significant ? 'SIGNIFICANT' : 'NOT SIGNIFICANT'}
                                        </Tag>
                                    </Card>
                                </Col>
                            </Row>
                        )}
                        {!evaluationResult.ready_for_decision && Array.isArray(evaluationResult.blockers) && evaluationResult.blockers.length > 0 && (
                            <div style={{ marginTop: 16 }}>
                                <Space wrap>
                                    {evaluationResult.blockers.map((b: string) => (
                                        <Tag key={b} color="orange">{String(b).replace(/_/g, ' ')}</Tag>
                                    ))}
                                </Space>
                            </div>
                        )}
                        <Text type="secondary" style={{ display: 'block', marginTop: 12 }}>
                            {evaluationResult.message || 'Evaluation complete.'}
                        </Text>
                        {evaluationUpdatedAt && (
                            <Text type="secondary" style={{ display: 'block', marginTop: 4 }}>
                                Last updated: {evaluationUpdatedAt}
                            </Text>
                        )}
                        <div style={{ marginTop: 12 }}>
                            <Button
                                size="small"
                                loading={evaluationLoading}
                                disabled={!evaluateModal}
                                onClick={() => evaluateModal && fetchEvaluation(evaluateModal)}
                            >
                                Refresh Evaluation
                            </Button>
                        </div>
                    </div>
                )}
            </Modal>

            <Modal
                title="Simulation In Progress"
                open={!!simulationInFlightTestId}
                footer={null}
                closable={false}
                maskClosable={false}
                keyboard={false}
            >
                <div style={{ paddingTop: 8 }}>
                    <Spin size="large" />
                    <div style={{ marginTop: 16 }}>
                        <Text strong>Running A/B simulation...</Text>
                        <br />
                        <Text type="secondary">
                            We are replaying transactions and computing metrics for champion and challenger.
                            This can take some time for larger datasets.
                        </Text>
                        <br />
                        <Text type="secondary">
                            Phase: {simulationProgress?.phase || 'INITIALIZING'}
                        </Text>
                        <br />
                        <Text type="secondary">
                            Processed: {Number(simulationProgress?.processed || 0).toLocaleString()}
                            {Number(simulationProgress?.total || 0) > 0
                                ? ` / ${Number(simulationProgress?.total || 0).toLocaleString()}`
                                : ''}
                        </Text>
                    </div>
                    <Progress
                        status="active"
                        percent={Number(simulationProgress?.percent || 0)}
                        showInfo
                        format={(percent) => `${Number(percent || 0).toFixed(1)}%`}
                        style={{ marginTop: 20 }}
                    />
                    {simulationProgress?.message && (
                        <Text type="danger">{simulationProgress.message}</Text>
                    )}
                </div>
            </Modal>

            {/* Simulation Modal */}
            <Modal
                title="Simulate A/B Traffic"
                open={!!simulateModal}
                onCancel={() => setSimulateModal(null)}
                onOk={() => simulateForm.submit()}
                confirmLoading={simulateMutation.isPending}
            >
                <div style={{ marginBottom: 16 }}>
                    <Text type="secondary">
                        This will replay historical transactions from your selected dataset through the A/B test router to generate performance metrics.
                    </Text>
                </div>
                <Form
                    form={simulateForm}
                    layout="vertical"
                    onFinish={(values) =>
                        simulateModal &&
                        simulateMutation.mutate({ testId: simulateModal, reset_existing: true, ...values })
                    }
                >
                    <Form.Item
                        name="dataset_id"
                        label="Source Dataset"
                        rules={[{ required: true, message: 'Please select a dataset' }]}
                    >
                        <Select placeholder="Select dataset">
                            {simulationEligibleDatasets.map((d: any) => (
                                <Option key={d.id} value={d.id}>
                                    {`${d.name} (${d.row_count} rows)`}
                                </Option>
                            ))}
                            {simulationEligibleDatasets.length === 0 && (
                                <Option key="__no_labelled_dataset__" value="__no_labelled_dataset__" disabled>
                                    No labelled dataset available
                                </Option>
                            )}
                        </Select>
                    </Form.Item>
                    <Form.Item
                        name="rows"
                        label="Number of Transactions"
                        initialValue={1000}
                    >
                        <InputNumber min={1} style={{ width: '100%' }} />
                    </Form.Item>
                </Form>
            </Modal>
        </div>
    );
}