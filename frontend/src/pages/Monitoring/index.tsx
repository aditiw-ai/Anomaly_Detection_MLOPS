/**
 * Monitoring Page
 * Track data drift, model performance, anomaly detection health, and bias metrics.
 */

import { useState, useEffect } from 'react';
import {
    Card,
    Row,
    Col,
    Typography,
    Tag,
    Table,
    Tabs,
    Progress,
    Statistic,
    Space,
    Button,
    Tooltip,
    Alert,
    Select,
    Skeleton,
} from 'antd';
import {
    LineChartOutlined,
    WarningOutlined,
    CheckCircleOutlined,
    ReloadOutlined,
    ExclamationCircleOutlined,
} from '@ant-design/icons';
import {
    useQuery,
    useMutation,
    useQueryClient,
} from '@tanstack/react-query';
import { monitoringService } from '@/services/monitoringService';

import { api } from '../../api/axios';

const { Title, Text } = Typography;

export function Monitoring() {
    const [activeTab, setActiveTab] = useState('drift');
    const [modelId, setModelId] = useState<string | null>(null);

    const [models, setModels] = useState<{
        model_id: string;
        name?: string;
        algorithm?: string;
        status?: string;
        metrics?: Record<string, number>;
    }[]>([]);

    const [modelsLoading, setModelsLoading] = useState(true);

    const queryClient = useQueryClient();

    // Fetch registered models
    useEffect(() => {
        const fetchModels = async () => {
            try {
                setModelsLoading(true);

                const response = await api.get(
                    '/inference/models'
                );

                const modelList =
                    response.data?.data ||
                    response.data ||
                    [];

                setModels(modelList);

                if (
                    modelList.length > 0 &&
                    !modelId
                ) {
                    setModelId(
                        modelList[0].model_id
                    );
                }
            } catch (error) {
                console.error(
                    'Failed to load models:',
                    error
                );
            } finally {
                setModelsLoading(false);
            }
        };

        fetchModels();
    }, []);

    // Fetch drift metrics
    const {
        data: driftData,
        isLoading: driftLoading,
    } = useQuery({
        queryKey: ['drift', modelId],
        queryFn: () =>
            monitoringService.getDriftMetrics(
                modelId!
            ),
        enabled: !!modelId,
        refetchInterval: 30000,
    });

    // Fetch bias metrics
    const {
        data: biasData,
        isLoading: biasLoading,
    } = useQuery({
        queryKey: ['bias', modelId],
        queryFn: () =>
            monitoringService.getBiasMetrics(
                modelId!
            ),
        enabled: !!modelId,
        refetchInterval: 30000,
    });

    // Fetch performance metrics
    const {
        data: performanceData,
        isLoading: performanceLoading,
    } = useQuery({
        queryKey: ['performance', modelId],
        queryFn: () =>
            monitoringService.getPerformanceMetrics(
                modelId!
            ),
        enabled: !!modelId,
        refetchInterval: 30000,
    });

    // Fetch baselines
    const {
        data: baselinesData,
        isLoading: baselinesLoading,
    } = useQuery({
        queryKey: ['baselines', modelId],
        queryFn: () =>
            monitoringService.getBaselines(
                modelId!
            ),
        enabled: !!modelId,
    });

    // Trigger monitoring computation
    const computeMutation = useMutation({
        mutationFn: async () => {
            if (!modelId) {
                throw new Error(
                    'Please select a model'
                );
            }

            await Promise.all([
                monitoringService.triggerDriftComputation(
                    modelId
                ),
                monitoringService.triggerBiasComputation(
                    modelId
                ),
            ]);
        },

        onSuccess: () => {
            queryClient.invalidateQueries({
                queryKey: ['drift'],
            });

            queryClient.invalidateQueries({
                queryKey: ['bias'],
            });

            queryClient.invalidateQueries({
                queryKey: ['performance'],
            });
        },
    });

    // Prepare drift table data
    const getDriftTableData = () => {
        if (
            !driftData?.data?.features
        ) {
            return [];
        }

        return Object.entries(
            driftData.data.features
        ).map(
            ([
                name,
                metrics,
            ]: [string, any]) => ({
                key: name,
                feature: name,
                psi: metrics.psi,
                ks: metrics.ks_statistic,
                pValue: metrics.p_value,
                status:
                    metrics.status ||
                    'OK',
            })
        );
    };

    // Prepare bias table data
    const getBiasTableData = () => {
        if (
            !biasData?.data
                ?.protected_attributes
        ) {
            return [];
        }

        return Object.entries(
            biasData.data
                .protected_attributes
        ).map(
            ([
                name,
                metrics,
            ]: [string, any]) => ({
                key: name,
                attribute: name,
                demographicParityDiff:
                    metrics.demographic_parity_diff,
                disparateImpact:
                    metrics.disparate_impact,
                status:
                    metrics.status ||
                    'OK',
            })
        );
    };

    const driftColumns = [
        {
            title: 'Feature',
            dataIndex: 'feature',
            key: 'feature',

            render: (
                name: string
            ) => (
                <Text strong>
                    {name}
                </Text>
            ),
        },

        {
            title: 'PSI',
            dataIndex: 'psi',
            key: 'psi',

            render: (
                value: number
            ) => {
                const psi =
                    Number(value) || 0;

                return (
                    <Tooltip
                        title={`PSI: ${psi.toFixed(
                            4
                        )}`}
                    >
                        <Tag
                            color={
                                psi >
                                0.25
                                    ? 'red'
                                    : psi >
                                      0.1
                                    ? 'orange'
                                    : 'green'
                            }
                        >
                            {psi.toFixed(4)}
                        </Tag>
                    </Tooltip>
                );
            },
        },

        {
            title: 'KS Statistic',
            dataIndex: 'ks',
            key: 'ks',

            render: (
                value: number
            ) =>
                Number(value)
                    .toFixed(4),
        },

        {
            title: 'P-Value',
            dataIndex: 'pValue',
            key: 'pValue',

            render: (
                value: number
            ) =>
                Number(value)
                    .toFixed(4),
        },

        {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',

            render: (
                status: string
            ) => (
                <Tag
                    color={
                        status ===
                        'CRITICAL'
                            ? 'red'
                            : status ===
                              'WARNING'
                            ? 'orange'
                            : 'green'
                    }
                >
                    {status}
                </Tag>
            ),
        },
    ];

    const biasColumns = [
        {
            title: 'Protected Attribute',
            dataIndex: 'attribute',
            key: 'attribute',

            render: (
                name: string
            ) => (
                <Text strong>
                    {name}
                </Text>
            ),
        },

        {
            title: (
                <Tooltip title="Difference in anomaly prediction rate between the most- and least-flagged demographic group. 0 = perfectly fair. > 0.1 = WARNING, > 0.2 = CRITICAL.">
                    <span
                        style={{
                            borderBottom:
                                '1px dashed #aaa',
                            cursor: 'help',
                        }}
                    >
                        Demographic Parity Diff ⓘ
                    </span>
                </Tooltip>
            ),

            dataIndex:
                'demographicParityDiff',

            key: 'demographicParityDiff',

            render: (
                value: number
            ) => {
                const diff =
                    Number(value) || 0;

                return (
                    <Tag
                        color={
                            diff >
                            0.2
                                ? 'red'
                                : diff >
                                  0.1
                                ? 'orange'
                                : 'green'
                        }
                    >
                        {diff.toFixed(
                            3
                        )}{' '}
                        {diff >
                        0.2
                            ? '⚠ HIGH'
                            : diff >
                              0.1
                            ? '⚠'
                            : '✓'}
                    </Tag>
                );
            },
        },

        {
            title: (
                <Tooltip title="Ratio of anomaly prediction rate of the least-flagged group vs most-flagged group. 80% Rule: must be ≥ 0.80. Values near 0% mean one group is almost never flagged while another is heavily flagged.">
                    <span
                        style={{
                            borderBottom:
                                '1px dashed #aaa',
                            cursor: 'help',
                        }}
                    >
                        Disparate Impact ⓘ
                    </span>
                </Tooltip>
            ),

            dataIndex:
                'disparateImpact',

            key: 'disparateImpact',

            render: (
                value: number
            ) => {
                const impact =
                    Number(value) || 0;

                return (
                    <Tag
                        color={
                            impact <
                            0.8
                                ? 'red'
                                : 'green'
                        }
                    >
                        {(
                            impact * 100
                        ).toFixed(
                            1
                        )}
                        %
                    </Tag>
                );
            },
        },

        {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',

            render: (
                status: string
            ) => (
                <Tag
                    color={
                        status ===
                        'CRITICAL'
                            ? 'red'
                            : status ===
                              'WARNING'
                            ? 'orange'
                            : 'green'
                    }
                >
                    {status}
                </Tag>
            ),
        },
    ];

    const getStatusIcon = (
        status: string
    ) => {
        if (
            status === 'OK'
        ) {
            return (
                <CheckCircleOutlined />
            );
        }

        if (
            status ===
            'CRITICAL'
        ) {
            return (
                <ExclamationCircleOutlined />
            );
        }

        if (
            status ===
            'WARNING'
        ) {
            return (
                <WarningOutlined />
            );
        }

        return null;
    };

    const performanceMetrics = [
        {
            key: 'precision',
            label: 'Precision',
        },
        {
            key: 'recall',
            label: 'Recall',
        },
        {
            key: 'f1',
            label: 'F1 Score',
        },
        {
            key: 'auc',
            label: 'AUC',
        },
    ];

    return (
        <div className="fade-in">
            {/* Header */}
            <div className="page-header">
                <div>
                    <Title
                        level={2}
                        style={{
                            margin: 0,
                        }}
                    >
                        Monitoring
                    </Title>

                    <Text type="secondary">
                        Track anomaly detection drift,
                        performance, and fairness
                    </Text>
                </div>

                <Space>
                    <Select
                        loading={
                            modelsLoading
                        }
                        value={
                            modelId ||
                            undefined
                        }
                        onChange={
                            setModelId
                        }
                        placeholder="Select model"
                        style={{
                            minWidth: 240,
                        }}
                        options={models.map(
                            (model) => ({
                                value:
                                    model.model_id,
                                label:
                                    model.name ||
                                    model.model_id,
                            })
                        )}
                    />

                    <Button
                        icon={
                            <ReloadOutlined />
                        }
                        loading={
                            computeMutation.isPending
                        }
                        disabled={
                            !modelId
                        }
                        onClick={() =>
                            computeMutation.mutate()
                        }
                    >
                        Refresh Monitoring
                    </Button>
                </Space>
            </div>

            {/* Monitoring Summary */}
            {modelId && (
                <Row
                    gutter={16}
                    style={{
                        marginBottom: 24,
                    }}
                >
                    <Col
                        xs={24}
                        md={8}
                    >
                        <Card>
                            <Statistic
                                title="Drift Status"
                                value={
                                    driftData
                                        ?.data
                                        ?.overall_status ===
                                    'NO_DATA'
                                        ? 'N/A'
                                        : driftData
                                              ?.data
                                              ?.overall_status ||
                                          'N/A'
                                }
                                prefix={getStatusIcon(
                                    driftData
                                        ?.data
                                        ?.overall_status ||
                                        ''
                                )}
                                valueStyle={{
                                    color:
                                        driftData
                                            ?.data
                                            ?.overall_status ===
                                        'OK'
                                            ? '#3f8600'
                                            : driftData
                                                  ?.data
                                                  ?.overall_status ===
                                              'WARNING'
                                            ? '#cf9700'
                                            : driftData
                                                  ?.data
                                                  ?.overall_status ===
                                              'CRITICAL'
                                            ? '#cf1322'
                                            : '#888',
                                }}
                            />
                        </Card>
                    </Col>

                    <Col
                        xs={24}
                        md={8}
                    >
                        <Card>
                            <Statistic
                                title="Bias Status"
                                value={
                                    biasData
                                        ?.data
                                        ?.overall_status ===
                                    'NO_DATA'
                                        ? 'N/A'
                                        : biasData
                                              ?.data
                                              ?.overall_status ||
                                          'N/A'
                                }
                                prefix={getStatusIcon(
                                    biasData
                                        ?.data
                                        ?.overall_status ||
                                        ''
                                )}
                                valueStyle={{
                                    color:
                                        biasData
                                            ?.data
                                            ?.overall_status ===
                                        'OK'
                                            ? '#3f8600'
                                            : biasData
                                                  ?.data
                                                  ?.overall_status ===
                                              'NO_DATA'
                                            ? '#888'
                                            : '#cf9700',
                                }}
                            />
                        </Card>
                    </Col>

                    <Col
                        xs={24}
                        md={8}
                    >
                        <Card>
                            <Statistic
                                title="Current F1 Score"
                                value={
                                    (performanceData
                                        ?.data
                                        ?.current
                                        ?.f1 ||
                                        0) *
                                    100
                                }
                                suffix="%"
                                precision={1}
                            />
                        </Card>
                    </Col>
                </Row>
            )}

            {/* Monitoring Tabs */}
            <Card>
                <Tabs
                    activeKey={
                        activeTab
                    }
                    onChange={
                        setActiveTab
                    }
                    items={[
                        {
                            key: 'drift',

                            label: (
                                <span>
                                    <LineChartOutlined />{' '}
                                    Data Drift
                                </span>
                            ),

                            children: (
                                <>
                                    {driftData
                                        ?.data
                                        ?.overall_status ===
                                        'CRITICAL' && (
                                        <Alert
                                            message="Critical Drift Detected"
                                            description="Significant drift has been detected in one or more features used by the anomaly detection model. Consider investigating the affected data and retraining the model if necessary."
                                            type="error"
                                            showIcon
                                            style={{
                                                marginBottom: 16,
                                            }}
                                        />
                                    )}

                                    <Table
                                        loading={
                                            driftLoading
                                        }
                                        dataSource={getDriftTableData()}
                                        columns={
                                            driftColumns
                                        }
                                        rowKey="key"
                                        pagination={{
                                            pageSize: 10,
                                        }}
                                        locale={{
                                            emptyText:
                                                'No drift data available',
                                        }}
                                    />
                                </>
                            ),
                        },

                        {
                            key: 'performance',

                            label: (
                                <span>
                                    <LineChartOutlined />{' '}
                                    Performance
                                </span>
                            ),

                            children: (
                                <Card title="Current Performance vs Baselines">
                                    {performanceLoading ||
                                    baselinesLoading ? (
                                        <Skeleton
                                            active
                                        />
                                    ) : (
                                        <Row
                                            gutter={[
                                                16,
                                                16,
                                            ]}
                                        >
                                            {performanceMetrics.map(
                                                (
                                                    metric
                                                ) => {
                                                    const currentVal =
                                                        (performanceData
                                                            ?.data
                                                            ?.current as any)?.[
                                                            metric.key
                                                        ] ||
                                                        0;

                                                    const baselineVal =
                                                        (baselinesData
                                                            ?.data as any)?.[
                                                            metric.key
                                                        ] ||
                                                        0;

                                                    const percent =
                                                        currentVal *
                                                        100;

                                                    return (
                                                        <Col
                                                            xs={
                                                                24
                                                            }
                                                            sm={
                                                                12
                                                            }
                                                            lg={
                                                                6
                                                            }
                                                            key={
                                                                metric.key
                                                            }
                                                        >
                                                            <Card
                                                                size="small"
                                                            >
                                                                <Statistic
                                                                    title={
                                                                        metric.label
                                                                    }
                                                                    value={
                                                                        percent
                                                                    }
                                                                    suffix="%"
                                                                    precision={
                                                                        1
                                                                    }
                                                                />

                                                                <Progress
                                                                    percent={
                                                                        Math.min(
                                                                            100,
                                                                            Math.max(
                                                                                0,
                                                                                percent
                                                                            )
                                                                        )
                                                                    }
                                                                    status={
                                                                        baselineVal &&
                                                                        currentVal <
                                                                            baselineVal
                                                                            ? 'exception'
                                                                            : 'normal'
                                                                    }
                                                                    showInfo={
                                                                        false
                                                                    }
                                                                />

                                                                <Text
                                                                    type="secondary"
                                                                    style={{
                                                                        fontSize: 12,
                                                                    }}
                                                                >
                                                                    Baseline:{' '}
                                                                    {(
                                                                        baselineVal *
                                                                        100
                                                                    ).toFixed(
                                                                        1
                                                                    )}
                                                                    %
                                                                </Text>
                                                            </Card>
                                                        </Col>
                                                    );
                                                }
                                            )}
                                        </Row>
                                    )}

                                    {performanceData
                                        ?.data
                                        ?.current && (
                                        <div
                                            style={{
                                                marginTop: 16,
                                            }}
                                        >
                                            <Text type="secondary">
                                                Performance metrics
                                                show how reliably the
                                                anomaly detection model
                                                distinguishes normal
                                                records from anomalous
                                                records.
                                            </Text>
                                        </div>
                                    )}
                                </Card>
                            ),
                        },

                        {
                            key: 'bias',

                            label: (
                                <span>
                                    <WarningOutlined />{' '}
                                    Bias Detection
                                </span>
                            ),

                            children: (
                                <>
                                    <Card
                                        title="Bias / Fairness Detection"
                                        style={{
                                            marginBottom: 16,
                                        }}
                                    >
                                        <Text type="secondary">
                                            Monitor whether anomaly
                                            predictions differ
                                            significantly across
                                            protected demographic
                                            groups.
                                        </Text>
                                    </Card>

                                    {biasData
                                        ?.data
                                        ?.overall_status ===
                                        'CRITICAL' && (
                                        <Alert
                                            message="Critical Bias Detected"
                                            description="One or more protected attributes show a severe disparity in how the model assigns anomaly labels. Review the attribute rows below and consider retraining with balanced data."
                                            type="error"
                                            showIcon
                                            style={{
                                                marginBottom: 16,
                                            }}
                                        />
                                    )}

                                    {biasData
                                        ?.data
                                        ?.overall_status ===
                                        'WARNING' && (
                                        <Alert
                                            message="Bias Warning"
                                            description="Fairness thresholds have been exceeded for one or more protected attributes."
                                            type="warning"
                                            showIcon
                                            style={{
                                                marginBottom: 16,
                                            }}
                                        />
                                    )}

                                    <Table
                                        loading={
                                            biasLoading
                                        }
                                        dataSource={getBiasTableData()}
                                        columns={
                                            biasColumns
                                        }
                                        rowKey="key"
                                        pagination={{
                                            pageSize: 10,
                                        }}
                                        locale={{
                                            emptyText:
                                                'No bias monitoring data available',
                                        }}
                                    />

                                    <Card
                                        size="small"
                                        style={{
                                            marginTop: 16,
                                        }}
                                    >
                                        <Text
                                            type="secondary"
                                            style={{
                                                fontSize: 12,
                                            }}
                                        >
                                            <strong>
                                                How to interpret:
                                            </strong>
                                            &nbsp;
                                            <strong>
                                                Demographic Parity Diff
                                            </strong>{' '}
                                            — difference in anomaly
                                            flagging rate between
                                            groups (lower is fairer,
                                            threshold: &lt;0.1).
                                            &nbsp;
                                            <strong>
                                                Disparate Impact
                                            </strong>{' '}
                                            — ratio of minimum group
                                            rate ÷ maximum group rate
                                            (higher is fairer, must be
                                            ≥80% per the 80% rule).
                                            &nbsp;
                                            A value of{' '}
                                            <strong>
                                                0.0%
                                            </strong>{' '}
                                            means one group is{' '}
                                            <em>
                                                never
                                            </em>{' '}
                                            predicted as anomalous
                                            while another is — a very
                                            strong signal.
                                        </Text>
                                    </Card>
                                </>
                            ),
                        },
                    ]}
                />
            </Card>
        </div>
    );
}