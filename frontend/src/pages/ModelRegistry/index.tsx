/**
 * Model Registry Page
 * View, compare, and promote anomaly detection models.
 */

import { useState } from 'react';
import {
    Card,
    Table,
    Button,
    Tag,
    Modal,
    Typography,
    Row,
    Col,
    Space,
    Statistic,
    Descriptions,
    Tooltip,
    message,
    Empty,
    Spin,
} from 'antd';
import {
    RocketOutlined,
    EyeOutlined,
    CheckCircleOutlined,
    StarFilled,
    DeleteOutlined,
} from '@ant-design/icons';
import {
    useQuery,
    useMutation,
    useQueryClient,
} from '@tanstack/react-query';
import {
    modelService,
    MLModel,
} from '@/services/modelService';
import { useAuth } from '@/contexts/AuthContext';

const { Title, Text } = Typography;

export function ModelRegistry() {
    const { hasRole } = useAuth();
    const canDeploy = hasRole(['ADMIN', 'DEPLOYER']);

    const [selectedModel, setSelectedModel] =
        useState<MLModel | null>(null);

    const [detailModalOpen, setDetailModalOpen] =
        useState(false);

    const queryClient = useQueryClient();

    // Fetch all registered models
    const {
        data: models,
        isLoading,
    } = useQuery({
        queryKey: ['models'],
        queryFn: () => modelService.listModels(),
        refetchInterval: 5000,
    });

    // Fetch current production model
    const { data: productionModel } = useQuery({
        queryKey: ['productionModel'],
        queryFn: () => modelService.getProductionModel(),
        refetchInterval: 5000,
    });

    // Fetch selected model details
    const {
        data: selectedModelDetails,
        isLoading: isDetailsLoading,
    } = useQuery({
        queryKey: ['model', selectedModel?.id],
        queryFn: () =>
            modelService.getModel(selectedModel!.id),
        enabled:
            !!selectedModel?.id &&
            detailModalOpen,
    });

    const displayModel =
        selectedModelDetails || selectedModel;

    // Determine whether the model belongs to Stage 1 or Stage 2
    const getModelStage = (model: MLModel) => {
        const trainingStage =
            model.hyperparameters?.training_stage;

        if (
            trainingStage === 'anomaly_type' ||
            trainingStage === 'stage_2' ||
            trainingStage === 'classification'
        ) {
            return 'Stage 2';
        }

        return 'Stage 1';
    };

    // Promote mutation
    const promoteMutation = useMutation({
        mutationFn: ({
            modelId,
            status,
        }: {
            modelId: string;
            status: string;
        }) =>
            modelService.promoteModel(
                modelId,
                status
            ),

        onSuccess: () => {
            queryClient.invalidateQueries({
                queryKey: ['models'],
            });

            queryClient.invalidateQueries({
                queryKey: ['productionModel'],
            });

            message.success(
                'Model promoted successfully'
            );
        },

        onError: (error: any) => {
            message.error(
                error.response?.data?.detail ||
                    'Failed to promote model'
            );
        },
    });

    // Delete mutation
    const deleteMutation = useMutation({
        mutationFn: (modelId: string) =>
            modelService.deleteModel(modelId),

        onSuccess: () => {
            queryClient.invalidateQueries({
                queryKey: ['models'],
            });

            queryClient.invalidateQueries({
                queryKey: ['productionModel'],
            });

            message.success(
                'Model deleted successfully'
            );
        },

        onError: (error: any) => {
            message.error(
                error.response?.data?.detail ||
                    'Failed to delete model'
            );
        },
    });

    const handleViewDetails = (
        model: MLModel
    ) => {
        setSelectedModel(model);
        setDetailModalOpen(true);
    };

    const handlePromote = (
        modelId: string,
        targetStatus: string
    ) => {
        promoteMutation.mutate({
            modelId,
            status: targetStatus,
        });
    };

    const handleDelete = (
        modelId: string,
        modelName: string
    ) => {
        Modal.confirm({
            title: 'Delete Model',

            content: `Are you sure you want to delete "${modelName}"? This action cannot be undone and will remove all associated files from storage.`,

            okText: 'Delete',
            okType: 'danger',
            cancelText: 'Cancel',

            onOk: () => {
                deleteMutation.mutate(modelId);
            },
        });
    };

    const columns = [
        {
            title: 'Name',
            dataIndex: 'name',
            key: 'name',

            render: (
                name: string,
                record: MLModel
            ) => (
                <Space>
                    {record.status ===
                        'PRODUCTION' && (
                        <StarFilled
                            style={{
                                color: '#faad14',
                            }}
                        />
                    )}

                    <Text strong>
                        {name}
                    </Text>
                </Space>
            ),
        },

        {
            title: 'Stage',
            key: 'stage',

            render: (
                _: unknown,
                record: MLModel
            ) => {
                const stage =
                    getModelStage(record);

                return (
                    <Tag
                        color={
                            stage === 'Stage 2'
                                ? 'purple'
                                : 'blue'
                        }
                    >
                        {stage}
                    </Tag>
                );
            },
        },

        {
            title: 'Version',
            dataIndex: 'version',
            key: 'version',

            render: (
                version: string
            ) => (
                <Tag>
                    {version}
                </Tag>
            ),
        },

        {
            title: 'Algorithm',
            dataIndex: 'algorithm',
            key: 'algorithm',

            render: (
                algo: string
            ) =>
                algo?.toUpperCase() ||
                '-',
        },

        {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',

            render: (
                status: string
            ) => {
                const colors: Record<
                    string,
                    string
                > = {
                    TRAINED: 'default',
                    STAGING: 'blue',
                    PRODUCTION: 'green',
                    ARCHIVED: 'gray',
                };

                return (
                    <Tag
                        color={
                            colors[status] ||
                            'default'
                        }
                    >
                        {status}
                    </Tag>
                );
            },
        },

        {
            title: 'Metrics',
            key: 'metrics',

            render: (
                _: unknown,
                record: MLModel
            ) => {
                if (!record.metrics) {
                    return '-';
                }

                return (
                    <Space>
                        <Tooltip title="Precision">
                            <Tag>
                                P:{' '}
                                {(
                                    record.metrics
                                        .precision *
                                    100
                                ).toFixed(1)}
                                %
                            </Tag>
                        </Tooltip>

                        <Tooltip title="Recall">
                            <Tag>
                                R:{' '}
                                {(
                                    record.metrics
                                        .recall *
                                    100
                                ).toFixed(1)}
                                %
                            </Tag>
                        </Tooltip>

                        <Tooltip title="F1 Score">
                            <Tag color="blue">
                                F1:{' '}
                                {(
                                    record.metrics
                                        .f1 *
                                    100
                                ).toFixed(1)}
                                %
                            </Tag>
                        </Tooltip>
                    </Space>
                );
            },
        },

        {
            title: 'Created',
            dataIndex: 'created_at',
            key: 'created_at',

            render: (
                date: string
            ) =>
                new Date(
                    date
                ).toLocaleDateString(),
        },

        {
            title: 'Actions',
            key: 'actions',

            render: (
                _: unknown,
                record: MLModel
            ) => (
                <Space>
                    <Button
                        size="small"
                        icon={
                            <EyeOutlined />
                        }
                        onClick={() =>
                            handleViewDetails(
                                record
                            )
                        }
                    >
                        Details
                    </Button>

                    {record.status ===
                        'TRAINED' && (
                        <Tooltip
                            title={
                                !canDeploy
                                    ? 'Your role does not have permission to deploy models.'
                                    : ''
                            }
                        >
                            <Button
                                size="small"
                                type="primary"
                                icon={
                                    <RocketOutlined />
                                }
                                onClick={() =>
                                    handlePromote(
                                        record.id,
                                        'STAGING'
                                    )
                                }
                                disabled={
                                    !canDeploy
                                }
                            >
                                Stage
                            </Button>
                        </Tooltip>
                    )}

                    {record.status ===
                        'STAGING' && (
                        <Tooltip
                            title={
                                !canDeploy
                                    ? 'Your role does not have permission to deploy models.'
                                    : ''
                            }
                        >
                            <Button
                                size="small"
                                type="primary"
                                icon={
                                    <CheckCircleOutlined />
                                }
                                onClick={() =>
                                    handlePromote(
                                        record.id,
                                        'PRODUCTION'
                                    )
                                }
                                disabled={
                                    !canDeploy
                                }
                            >
                                Deploy
                            </Button>
                        </Tooltip>
                    )}

                    <Tooltip
                        title={
                            !canDeploy
                                ? 'Your role does not have permission to delete models.'
                                : ''
                        }
                    >
                        <Button
                            size="small"
                            danger
                            icon={
                                <DeleteOutlined />
                            }
                            onClick={() =>
                                handleDelete(
                                    record.id,
                                    record.name
                                )
                            }
                            disabled={
                                !canDeploy ||
                                record.status ===
                                    'PRODUCTION'
                            }
                        >
                            Delete
                        </Button>
                    </Tooltip>
                </Space>
            ),
        },
    ];

    // Prepare feature importance data
    const getFeatureImportanceData = (
        model: MLModel | null
    ) => {
        if (!model?.feature_importance) {
            return [];
        }

        return Object.entries(
            model.feature_importance
        )
            .sort(
                ([, a], [, b]) =>
                    (b as number) -
                    (a as number)
            )
            .slice(0, 10)
            .map(
                ([name, value]) => ({
                    name,
                    importance: (
                        (value as number) *
                        100
                    ).toFixed(2),
                })
            );
    };

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
                        Model Registry
                    </Title>

                    <Text type="secondary">
                        View, compare, and deploy
                        anomaly detection models
                    </Text>
                </div>
            </div>

            {/* Production Model Card */}
            {productionModel?.data && (
                <Card
                    style={{
                        marginBottom: 24,
                        borderColor:
                            '#52c41a',
                    }}
                >
                    <Row
                        gutter={24}
                        align="middle"
                    >
                        <Col>
                            <StarFilled
                                style={{
                                    fontSize: 48,
                                    color:
                                        '#faad14',
                                }}
                            />
                        </Col>

                        <Col flex={1}>
                            <Title
                                level={4}
                                style={{
                                    margin: 0,
                                }}
                            >
                                Production Model:{' '}
                                {
                                    productionModel
                                        .data
                                        .name
                                }
                            </Title>

                            <Space>
                                <Text type="secondary">
                                    {
                                        productionModel
                                            .data
                                            .algorithm
                                    }{' '}
                                    • v
                                    {
                                        productionModel
                                            .data
                                            .version
                                    }
                                </Text>

                                <Tag
                                    color={
                                        getModelStage(
                                            productionModel
                                                .data
                                        ) ===
                                        'Stage 2'
                                            ? 'purple'
                                            : 'blue'
                                    }
                                >
                                    {
                                        getModelStage(
                                            productionModel
                                                .data
                                        )
                                    }
                                </Tag>
                            </Space>
                        </Col>

                        <Col>
                            <Space size="large">
                                <Statistic
                                    title="Precision"
                                    value={
                                        (
                                            productionModel
                                                .data
                                                .metrics
                                                ?.precision ||
                                            0
                                        ) * 100
                                    }
                                    suffix="%"
                                    precision={1}
                                />

                                <Statistic
                                    title="Recall"
                                    value={
                                        (
                                            productionModel
                                                .data
                                                .metrics
                                                ?.recall ||
                                            0
                                        ) * 100
                                    }
                                    suffix="%"
                                    precision={1}
                                />

                                <Statistic
                                    title="F1 Score"
                                    value={
                                        (
                                            productionModel
                                                .data
                                                .metrics
                                                ?.f1 ||
                                            0
                                        ) * 100
                                    }
                                    suffix="%"
                                    precision={1}
                                    valueStyle={{
                                        color:
                                            '#3f8600',
                                    }}
                                />
                            </Space>
                        </Col>
                    </Row>
                </Card>
            )}

            {/* Models Table */}
            <Card title="Registered Anomaly Models">
                <Table
                    loading={isLoading}
                    dataSource={
                        models?.data || []
                    }
                    columns={columns}
                    rowKey="id"
                    pagination={{
                        pageSize: 10,

                        showTotal: (
                            total: number
                        ) =>
                            `Total ${total} models`,
                    }}
                />
            </Card>

            {/* Model Detail Modal */}
            <Modal
                title={
                    displayModel?.name ||
                    'Model Details'
                }
                open={detailModalOpen}
                onCancel={() =>
                    setDetailModalOpen(false)
                }
                width={800}
                footer={null}
            >
                {isDetailsLoading ? (
                    <div
                        style={{
                            textAlign:
                                'center',
                            padding:
                                '40px 0',
                        }}
                    >
                        <Spin size="large" />
                    </div>
                ) : (
                    displayModel && (
                        <>
                            <Descriptions
                                bordered
                                size="small"
                                column={2}
                            >
                                <Descriptions.Item label="Stage">
                                    <Tag
                                        color={
                                            getModelStage(
                                                displayModel
                                            ) ===
                                            'Stage 2'
                                                ? 'purple'
                                                : 'blue'
                                        }
                                    >
                                        {
                                            getModelStage(
                                                displayModel
                                            )
                                        }
                                    </Tag>
                                </Descriptions.Item>

                                <Descriptions.Item label="Version">
                                    {
                                        displayModel.version
                                    }
                                </Descriptions.Item>

                                <Descriptions.Item label="Algorithm">
                                    {displayModel.algorithm?.toUpperCase()}
                                </Descriptions.Item>

                                <Descriptions.Item label="Status">
                                    <Tag>
                                        {
                                            displayModel.status
                                        }
                                    </Tag>
                                </Descriptions.Item>

                                <Descriptions.Item label="Created">
                                    {new Date(
                                        displayModel.created_at
                                    ).toLocaleString()}
                                </Descriptions.Item>
                            </Descriptions>

                            <Title
                                level={5}
                                style={{
                                    marginTop: 24,
                                }}
                            >
                                Performance Metrics
                            </Title>

                            <Row gutter={16}>
                                {displayModel.metrics &&
                                    Object.entries(
                                        displayModel.metrics
                                    ).map(
                                        ([
                                            key,
                                            value,
                                        ]) => (
                                            <Col
                                                span={
                                                    6
                                                }
                                                key={
                                                    key
                                                }
                                            >
                                                <Card
                                                    size="small"
                                                >
                                                    <Statistic
                                                        title={
                                                            key
                                                                .charAt(
                                                                    0
                                                                )
                                                                .toUpperCase() +
                                                            key.slice(
                                                                1
                                                            )
                                                        }
                                                        value={
                                                            (value as number) *
                                                            100
                                                        }
                                                        suffix="%"
                                                        precision={
                                                            2
                                                        }
                                                    />
                                                </Card>
                                            </Col>
                                        )
                                    )}
                            </Row>

                            <Title
                                level={5}
                                style={{
                                    marginTop: 24,
                                }}
                            >
                                Feature Importance (Top 10)
                            </Title>

                            {displayModel.feature_importance ? (
                                <Table
                                    dataSource={getFeatureImportanceData(
                                        displayModel
                                    )}
                                    pagination={false}
                                    size="middle"
                                    scroll={{
                                        y: 300,
                                    }}
                                    rowKey="name"
                                    columns={[
                                        {
                                            title: 'Feature',
                                            dataIndex:
                                                'name',
                                            key: 'name',

                                            render: (
                                                text: string
                                            ) => (
                                                <Text
                                                    style={{
                                                        fontSize:
                                                            '14px',
                                                        fontWeight:
                                                            500,
                                                    }}
                                                >
                                                    {
                                                        text
                                                    }
                                                </Text>
                                            ),
                                        },

                                        {
                                            title: 'Importance (%)',
                                            dataIndex:
                                                'importance',
                                            key: 'importance',
                                            width: 150,

                                            render: (
                                                val: string
                                            ) => (
                                                <Tag
                                                    color="blue"
                                                    style={{
                                                        fontSize:
                                                            '13px',
                                                        padding:
                                                            '2px 8px',
                                                    }}
                                                >
                                                    {
                                                        val
                                                    }
                                                    %
                                                </Tag>
                                            ),
                                        },
                                    ]}
                                />
                            ) : (
                                <Empty description="No feature importance data available" />
                            )}
                        </>
                    )
                )}
            </Modal>
        </div>
    );
}