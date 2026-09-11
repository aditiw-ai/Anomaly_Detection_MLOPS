/**
 * Dashboard Page
 * Overview of platform status and key metrics.
 */
import { useEffect, useState } from 'react';
import { Row, Col, Card, Statistic, Table, Tag, Typography, Tooltip, Select, Spin, Alert } from 'antd';
import {
    DatabaseOutlined,
    ExperimentOutlined,
    AppstoreOutlined,
    AlertOutlined,
    ArrowUpOutlined,
    ArrowDownOutlined,
    CheckCircleOutlined,
    CloseCircleOutlined,
} from '@ant-design/icons';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';
import { monitoringService, BaselineRecord } from '../../services/monitoringService';
import { alertService, Alert as AlertRecord } from '../../services/alertService';
import { api } from '../../api/axios';

const { Title, Text } = Typography;

const alertColumns = [
    { title: 'Type', dataIndex: 'alert_type', key: 'alert_type' },
    { title: 'Message', dataIndex: 'message', key: 'message' },
    {
        title: 'Severity',
        dataIndex: 'severity',
        key: 'severity',
        render: (severity: string) => {
            const colors: Record<string, string> = { CRITICAL: 'red', WARNING: 'orange', INFO: 'blue' };
            return <Tag color={colors[severity]}>{severity.toUpperCase()}</Tag>;
        },
    },
    {
        title: 'Created',
        dataIndex: 'created_at',
        key: 'created_at',
        render: (date: string) => date ? new Date(date).toLocaleString() : '-',
    },
];

interface ModelOption {
    model_id: string;
    label: string;
    metrics: Record<string, number>;
    status: string;
}

interface LiveMetrics {
    precision: number | null;
    recall: number | null;
    f1: number | null;
    auc: number | null;
    fpr: number | null;
}

function getBaseline(baselines: BaselineRecord[], metricName: string): BaselineRecord | undefined {
    return baselines.find(b => b.metric_name === metricName);
}

function metricPasses(value: number, baseline: BaselineRecord | undefined): boolean | null {
    if (!baseline) return null;
    if (baseline.operator === 'gte') return value >= baseline.threshold;
    if (baseline.operator === 'lte') return value <= baseline.threshold;
    return null;
}

interface DashboardStats {
    total_datasets: number | null;
    total_training_jobs: number | null;
    active_training_jobs: number | null;
    production_models: number | null;
    active_alerts: number | null;
}

export function Dashboard() {
    const [models, setModels] = useState<ModelOption[]>([]);
    const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
    const [liveMetrics, setLiveMetrics] = useState<LiveMetrics | null>(null);
    const [baselines, setBaselines] = useState<BaselineRecord[]>([]);
    const [trafficData, setTrafficData] = useState<any[]>([]);
    const [recentAlerts, setRecentAlerts] = useState<AlertRecord[]>([]);
    const [stats, setStats] = useState<DashboardStats>({
        total_datasets: null,
        total_training_jobs: null,
        active_training_jobs: null,
        production_models: null,
        active_alerts: null
    });
    const [statsLoading, setStatsLoading] = useState(true);
    const [statsError, setStatsError] = useState<string | null>(null);

    // Fetch live dashboard stats (global)
    useEffect(() => {
        let mounted = true;
        const watchdog = window.setTimeout(() => {
            if (!mounted) return;
            setStatsLoading(false);
            setStatsError(prev => prev || 'Dashboard metrics request timed out. Please refresh.');
        }, 16000);

        const fetchStats = async () => {
            try {
                const statsRes = await api.get(`/dashboard/stats`);
                const d = statsRes.data?.data;

                if (!mounted) return;
                setStats({
                    total_datasets: d?.total_datasets ?? null,
                    total_training_jobs: d?.total_training_jobs ?? null,
                    active_training_jobs: d?.active_training_jobs ?? null,
                    production_models: d?.production_models ?? null,
                    active_alerts: d?.active_alerts ?? null,
                });
                setStatsError(null);
            } catch {
                if (!mounted) return;
                setStats({
                    total_datasets: null,
                    total_training_jobs: null,
                    active_training_jobs: null,
                    production_models: null,
                    active_alerts: null
                });
                setStatsError('Unable to load dashboard metrics.');
            } finally {
                if (!mounted) return;
                setStatsLoading(false);
                window.clearTimeout(watchdog);
            }
        };
        fetchStats();

        return () => {
            mounted = false;
            window.clearTimeout(watchdog);
        };
    }, []);

    // Fetch Analytics Traffic (Filterable by Model)
    useEffect(() => {
        const fetchAnalytics = async () => {
            try {
                const url = selectedModelId
                    ? `/analytics/dashboard?model_id=${selectedModelId}`
                    : `/analytics/dashboard`;

                const analyticsRes = await api.get(url);
                const a = analyticsRes.data?.data;

                // Map API traffic response (date, request_count) to what Recharts expects
                if (a?.traffic) {
                    setTrafficData(a.traffic.map((t: any) => {
                        const date = new Date(t.date);
                        const name = isNaN(date.getTime()) 
                            ? 'Invalid Date' 
                            : date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                        return { name, predictions: t.request_count };
                    }));
                } else {
                    setTrafficData([]);
                }
            } catch {
                setTrafficData([]);
            }
        };

        fetchAnalytics();
    }, [selectedModelId]);

    // Initial load - fetch all models
    useEffect(() => {
        const load = async () => {
            try {
                const res = await api.get(`/inference/models`);
                const data: any[] = res.data?.data || [];
                if (!data.length) return;

                const options: ModelOption[] = data.map(m => {
                    const date = m.created_at ? new Date(m.created_at) : null;
                    const dateStr = date && !isNaN(date.getTime()) ? date.toLocaleString() : 'N/A';
                    return {
                        model_id: m.model_id,
                        label: `${m.name} - ${dateStr}`,
                        metrics: m.metrics || {},
                        status: m.status,
                    };
                });
                setModels(options);

                // Default: most recent model
                const first = options[0];
                setSelectedModelId(first.model_id);
                applyModel(first);
            } catch {
                // silently fail
            }
        };
        load();
    }, []);

    useEffect(() => {
        const loadAlerts = async () => {
            try {
                const res = await alertService.listAlerts();
                setRecentAlerts((res.data || []).slice(0, 5));
            } catch {
                setRecentAlerts([]);
            }
        };
        loadAlerts();
    }, []);

    const applyModel = async (model: ModelOption) => {
        const m = model.metrics;
        setLiveMetrics({
            precision: m.precision ?? null,
            recall: m.recall ?? null,
            f1: m.f1 ?? null,
            auc: m.auc ?? null,
            fpr: m.fpr ?? null,
        });
        try {
            const blRes = await monitoringService.getBaselines(model.model_id);
            setBaselines(blRes.data || []);
        } catch {
            setBaselines([]);
        }
    };

    const handleModelChange = (modelId: string) => {
        const selected = models.find(m => m.model_id === modelId);
        if (!selected) return;
        setSelectedModelId(modelId);
        applyModel(selected);
    };

    const getStageLabel = (model: ModelOption) => {
        const hyperparameters = (model as any).hyperparameters || {};
        if (hyperparameters.training_stage === 'anomaly_type') return 'Stage 2 anomaly-type classifier';
        return 'Stage 1 binary anomaly detector';
    };

    const renderMetric = (label: string, metricKey: keyof LiveMetrics, isLowerBetter = false) => {
        const raw = liveMetrics?.[metricKey];
        const value = raw != null ? +(raw * 100).toFixed(1) : null;
        const baseline = getBaseline(baselines, metricKey);
        const passed = raw != null ? metricPasses(raw, baseline) : null;

        const color = passed === true ? '#059669' : passed === false ? '#DC2626' : undefined;
        const statusIcon = passed === true
            ? <CheckCircleOutlined style={{ color: '#059669' }} />
            : passed === false
                ? <CloseCircleOutlined style={{ color: '#DC2626' }} />
                : (isLowerBetter ? <ArrowDownOutlined /> : <ArrowUpOutlined />);

        const thresholdLabel = baseline
            ? `Threshold: ${isLowerBetter ? '<=' : '>='} ${+(baseline.threshold * 100).toFixed(1)}%`
            : 'No baseline set';

        return (
            <Col span={12}>
                <Tooltip title={thresholdLabel}>
                    <Statistic
                        title={label}
                        value={value ?? '-'}
                        suffix={value != null ? '%' : ''}
                        prefix={statusIcon}
                        valueStyle={{ color, fontSize: 24 }}
                    />
                </Tooltip>
            </Col>
        );
    };

    return (
        <div className="fade-in">
            <div className="page-header">
                <Title level={2} style={{ margin: 0 }}>Dashboard</Title>
                <Text type="secondary">AnomaLens anomaly detection MLOps control center</Text>
            </div>

            {statsError && (
                <Alert
                    type="warning"
                    showIcon
                    style={{ marginBottom: 16 }}
                    message={statsError}
                />
            )}

            {/* Stats Cards */}
            <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                <Col xs={24} sm={12} lg={6}>
                    <Card>
                        <Spin spinning={statsLoading} size="small">
                            <Statistic
                                title="Datasets"
                                value={stats.total_datasets ?? '-'}
                                prefix={<DatabaseOutlined />}
                                valueStyle={{ color: '#2563EB' }}
                            />
                        </Spin>
                    </Card>
                </Col>
                <Col xs={24} sm={12} lg={6}>
                    <Card>
                        <Spin spinning={statsLoading} size="small">
                            <Statistic
                                title="Training Jobs"
                                value={stats.total_training_jobs ?? '-'}
                                prefix={<ExperimentOutlined />}
                                suffix={
                                    stats.active_training_jobs != null && stats.active_training_jobs > 0
                                        ? <Text type="secondary" style={{ fontSize: 14 }}>/ {stats.active_training_jobs} active</Text>
                                        : undefined
                                }
                            />
                        </Spin>
                    </Card>
                </Col>
                <Col xs={24} sm={12} lg={6}>
                    <Card>
                        <Spin spinning={statsLoading} size="small">
                            <Statistic
                                title="Production Models"
                                value={stats.production_models ?? '-'}
                                prefix={<AppstoreOutlined />}
                                valueStyle={{ color: '#059669' }}
                            />
                        </Spin>
                    </Card>
                </Col>
                <Col xs={24} sm={12} lg={6}>
                    <Card>
                        <Spin spinning={statsLoading} size="small">
                            <Statistic
                                title="Active Alerts"
                                value={stats.active_alerts ?? '-'}
                                prefix={<AlertOutlined />}
                                valueStyle={{ color: '#DC2626' }}
                            />
                        </Spin>
                    </Card>
                </Col>
            </Row>

            {/* Charts */}
            <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
                <Col xs={24} lg={16}>
                    <Card title="Inference Activity">
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={trafficData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis dataKey="name" />
                                <YAxis />
                                <RechartsTooltip />
                                <Line type="monotone" dataKey="predictions" name="Inference Requests" stroke="#2563EB" strokeWidth={2} />
                            </LineChart>
                        </ResponsiveContainer>
                    </Card>
                </Col>
                <Col xs={24} lg={8}>
                    <Card
                        title="Model Performance"
                        extra={
                            models.length > 0
                                ? <Tag color="green">Live</Tag>
                                : <Tag color="default">No model</Tag>
                        }
                    >
                        {/* Model selector */}
                        {models.length > 0 && (
                            <Select
                                value={selectedModelId ?? undefined}
                                onChange={handleModelChange}
                                style={{ width: '100%', marginBottom: 16 }}
                                size="small"
                                options={models.map(m => ({
                                    value: m.model_id,
                                    label: m.label,
                                }))}
                            />
                        )}
                        {selectedModelId && (
                            <Text type="secondary" style={{ display: 'block', marginBottom: 12 }}>
                                {getStageLabel(models.find(m => m.model_id === selectedModelId)!)}
                            </Text>
                        )}
                        <Row gutter={[16, 16]}>
                            {renderMetric('Precision', 'precision')}
                            {renderMetric('Recall', 'recall')}
                            {renderMetric('F1 Score', 'f1')}
                            {renderMetric('AUC-ROC', 'auc')}
                        </Row>
                    </Card>
                </Col>
            </Row>

            {/* Recent Alerts */}
            <Card title="Recent Anomaly Monitoring Alerts" extra={<a href="/alerts">View All</a>}>
                <Table
                    dataSource={recentAlerts}
                    columns={alertColumns}
                    rowKey="id"
                    pagination={false}
                    size="small"
                />
            </Card>
        </div>
    );
}

