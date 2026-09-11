/**
 * Model Comparison Page
 * Compare two models side by side.
 */
import { useState } from 'react';
import {
    Card, Row, Col, Typography, Select, Table, Tag,
    Statistic, Space, Empty
} from 'antd';
import {
    SwapOutlined, TrophyOutlined, ArrowUpOutlined, ArrowDownOutlined,
    MinusOutlined
} from '@ant-design/icons';
import { useQuery } from '@tanstack/react-query';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Legend, RadarChart, PolarGrid,
    PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import { modelService, MLModel } from '@/services/modelService';

const { Title, Text } = Typography;
const { Option } = Select;

export function ModelComparison() {
    const [modelAId, setModelAId] = useState<string | null>(null);
    const [modelBId, setModelBId] = useState<string | null>(null);

    // Fetch all models
    const { data: modelsData, isLoading: modelsLoading } = useQuery({
        queryKey: ['models'],
        queryFn: () => modelService.listModels(),
    });

    // Fetch comparison when both models selected
    const { data: comparisonData, isLoading: comparisonLoading } = useQuery({
        queryKey: ['comparison', modelAId, modelBId],
        queryFn: () => modelService.compareModels(modelAId!, modelBId!),
        enabled: !!modelAId && !!modelBId && modelAId !== modelBId,
    });

    const models = modelsData?.data || [];
    const comparison = comparisonData?.data;

    // Prepare radar chart data
    const getRadarData = () => {
        if (!comparison) return [];

        const metrics = ['precision', 'recall', 'f1', 'auc', 'accuracy'];
        return metrics.map(metric => {
            const m = comparison[metric];
            return {
                metric: metric.toUpperCase(),
                modelA: m?.[modelAId!] ? m[modelAId!] * 100 : 0,
                modelB: m?.[modelBId!] ? m[modelBId!] * 100 : 0,
            };
        });
    };

    // Prepare bar chart data
    const getBarData = () => {
        if (!comparison) return [];

        return Object.entries(comparison).map(([metric, values]: [string, any]) => ({
            metric,
            modelA: values[modelAId!] ? values[modelAId!] * 100 : 0,
            modelB: values[modelBId!] ? values[modelBId!] * 100 : 0,
            diff: values.difference ? values.difference * 100 : 0,
        }));
    };

    const getChangeIcon = (diff: number | null) => {
        if (diff === null || Math.abs(diff) < 0.01) {
            return <MinusOutlined style={{ color: '#8c8c8c' }} />;
        }
        return diff > 0
            ? <ArrowUpOutlined style={{ color: '#52c41a' }} />
            : <ArrowDownOutlined style={{ color: '#ff4d4f' }} />;
    };

    const comparisonColumns = [
        {
            title: 'Metric',
            dataIndex: 'metric',
            key: 'metric',
            render: (m: string) => <Text strong>{m.toUpperCase()}</Text>,
        },
        {
            title: modelAId ? models.find((m: MLModel) => m.id === modelAId)?.name || 'Model A' : 'Model A',
            dataIndex: 'modelA',
            key: 'modelA',
            render: (v: number) => `${v.toFixed(2)}%`,
        },
        {
            title: modelBId ? models.find((m: MLModel) => m.id === modelBId)?.name || 'Model B' : 'Model B',
            dataIndex: 'modelB',
            key: 'modelB',
            render: (v: number) => `${v.toFixed(2)}%`,
        },
        {
            title: 'Difference',
            dataIndex: 'diff',
            key: 'diff',
            render: (diff: number) => (
                <Space>
                    {getChangeIcon(diff)}
                    <Text type={diff > 0 ? 'success' : diff < 0 ? 'danger' : 'secondary'}>
                        {diff > 0 ? '+' : ''}{diff.toFixed(2)}%
                    </Text>
                </Space>
            ),
        },
        {
            title: 'Winner',
            key: 'winner',
            render: (_: any, record: any) => {
                if (Math.abs(record.diff) < 0.5) {
                    return <Tag>Tie</Tag>;
                }
                const winner = record.diff > 0 ? 'A' : 'B';
                return <Tag color="gold" icon={<TrophyOutlined />}>Model {winner}</Tag>;
            },
        },
    ];

    return (
        <div className="fade-in">
            <div className="page-header">
                <div>
                    <Title level={2} style={{ margin: 0 }}>Model Comparison</Title>
                    <Text type="secondary">Compare performance metrics between models</Text>
                </div>
            </div>

            {/* Model Selection */}
            <Card style={{ marginBottom: 24 }}>
                <Row gutter={24} align="middle">
                    <Col span={10}>
                        <Text strong>Model A</Text>
                        <Select
                            placeholder="Select first model"
                            style={{ width: '100%', marginTop: 8 }}
                            loading={modelsLoading}
                            value={modelAId}
                            onChange={setModelAId}
                        >
                            {models.map((m: MLModel) => (
                                <Option key={m.id} value={m.id} disabled={m.id === modelBId}>
                                    {m.name} (v{m.version})
                                </Option>
                            ))}
                        </Select>
                    </Col>
                    <Col span={4} style={{ textAlign: 'center' }}>
                        <SwapOutlined style={{ fontSize: 24, color: '#1890ff' }} />
                    </Col>
                    <Col span={10}>
                        <Text strong>Model B</Text>
                        <Select
                            placeholder="Select second model"
                            style={{ width: '100%', marginTop: 8 }}
                            loading={modelsLoading}
                            value={modelBId}
                            onChange={setModelBId}
                        >
                            {models.map((m: MLModel) => (
                                <Option key={m.id} value={m.id} disabled={m.id === modelAId}>
                                    {m.name} (v{m.version})
                                </Option>
                            ))}
                        </Select>
                    </Col>
                </Row>
            </Card>

            {/* Comparison Results */}
            {modelAId && modelBId && modelAId !== modelBId ? (
                <>
                    {/* Summary Cards */}
                    <Row gutter={16} style={{ marginBottom: 24 }}>
                        <Col span={8}>
                            <Card>
                                <Statistic
                                    title="Model A F1 Score"
                                    value={comparison?.f1?.[modelAId] ? comparison.f1[modelAId] * 100 : 0}
                                    suffix="%"
                                    precision={1}
                                />
                            </Card>
                        </Col>
                        <Col span={8}>
                            <Card>
                                <Statistic
                                    title="Model B F1 Score"
                                    value={comparison?.f1?.[modelBId] ? comparison.f1[modelBId] * 100 : 0}
                                    suffix="%"
                                    precision={1}
                                />
                            </Card>
                        </Col>
                        <Col span={8}>
                            <Card>
                                <Statistic
                                    title="F1 Difference"
                                    value={comparison?.f1?.difference ? comparison.f1.difference * 100 : 0}
                                    prefix={getChangeIcon(comparison?.f1?.difference || 0)}
                                    suffix="%"
                                    precision={2}
                                    valueStyle={{
                                        color: (comparison?.f1?.difference || 0) > 0 ? '#3f8600' : '#cf1322'
                                    }}
                                />
                            </Card>
                        </Col>
                    </Row>

                    {/* Charts */}
                    <Row gutter={24} style={{ marginBottom: 24 }}>
                        <Col span={12}>
                            <Card title="Metrics Comparison">
                                <ResponsiveContainer width="100%" height={300}>
                                    <BarChart data={getBarData()}>
                                        <CartesianGrid strokeDasharray="3 3" />
                                        <XAxis dataKey="metric" />
                                        <YAxis domain={[0, 100]} />
                                        <Tooltip formatter={(v: number) => `${v.toFixed(2)}%`} />
                                        <Legend />
                                        <Bar dataKey="modelA" fill="#2563EB" name="Model A" />
                                        <Bar dataKey="modelB" fill="#059669" name="Model B" />
                                    </BarChart>
                                </ResponsiveContainer>
                            </Card>
                        </Col>
                        <Col span={12}>
                            <Card title="Performance Radar">
                                <ResponsiveContainer width="100%" height={300}>
                                    <RadarChart data={getRadarData()}>
                                        <PolarGrid />
                                        <PolarAngleAxis dataKey="metric" />
                                        <PolarRadiusAxis domain={[0, 100]} />
                                        <Radar
                                            name="Model A"
                                            dataKey="modelA"
                                            stroke="#2563EB"
                                            fill="#2563EB"
                                            fillOpacity={0.3}
                                        />
                                        <Radar
                                            name="Model B"
                                            dataKey="modelB"
                                            stroke="#059669"
                                            fill="#059669"
                                            fillOpacity={0.3}
                                        />
                                        <Legend />
                                    </RadarChart>
                                </ResponsiveContainer>
                            </Card>
                        </Col>
                    </Row>

                    {/* Detailed Comparison Table */}
                    <Card title="Detailed Metrics">
                        <Table
                            loading={comparisonLoading}
                            dataSource={getBarData()}
                            columns={comparisonColumns}
                            rowKey="metric"
                            pagination={false}
                        />
                    </Card>
                </>
            ) : (
                <Card>
                    <Empty
                        description="Select two different models to compare"
                        image={Empty.PRESENTED_IMAGE_SIMPLE}
                    />
                </Card>
            )}
        </div>
    );
}
