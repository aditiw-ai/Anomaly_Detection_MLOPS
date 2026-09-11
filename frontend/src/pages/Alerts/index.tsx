 /**
  * Alerts Page
  * View and manage anomaly detection and MLOps system alerts.
  */

import { useState } from 'react';
import {
    Card,
    Table,
    Tag,
    Button,
    Typography,
    Row,
    Col,
    Space,
    Statistic,
    Modal,
    Input,
    message,
    Badge,
    Drawer,
    Descriptions,
    Divider,
} from 'antd';
import {
    CheckCircleOutlined,
    ClockCircleOutlined,
    ExclamationCircleOutlined,
    WarningOutlined,
    InfoCircleOutlined,
    RobotOutlined,
} from '@ant-design/icons';
import {
    useQuery,
    useMutation,
    useQueryClient,
} from '@tanstack/react-query';
import {
    alertService,
    Alert,
} from '@/services/alertService';

const { Title, Text } = Typography;

export function Alerts() {
    const [acknowledgeModal, setAcknowledgeModal] =
        useState(false);

    const [selectedAlert, setSelectedAlert] =
        useState<Alert | null>(null);

    const [detailDrawer, setDetailDrawer] =
        useState(false);

    const [resolutionNote, setResolutionNote] =
        useState('');

    const queryClient = useQueryClient();

    const {
        data: alertsData,
        isLoading,
    } = useQuery({
        queryKey: ['alerts'],
        queryFn: () =>
            alertService.listAlerts(),
    });

    const acknowledgeMutation = useMutation({
        mutationFn: ({
            alertId,
            note,
        }: {
            alertId: string;
            note?: string;
        }) =>
            alertService.acknowledgeAlert(
                alertId,
                note
            ),

        onSuccess: () => {
            queryClient.invalidateQueries({
                queryKey: ['alerts'],
            });

            setAcknowledgeModal(false);
            setDetailDrawer(false);
            setSelectedAlert(null);
            setResolutionNote('');

            message.success(
                'Alert acknowledged'
            );
        },
    });

    const resolveMutation = useMutation({
        mutationFn: ({
            alertId,
            note,
        }: {
            alertId: string;
            note?: string;
        }) =>
            alertService.resolveAlert(
                alertId,
                note
            ),

        onSuccess: () => {
            queryClient.invalidateQueries({
                queryKey: ['alerts'],
            });

            setDetailDrawer(false);

            message.success(
                'Alert resolved'
            );
        },
    });

    const isInformationalTrainingAlert = (
        alert: Alert | null
    ) =>
        !!alert &&
        alert.alert_type === 'TRAINING' &&
        alert.severity === 'INFO';

    const getSeverityIcon = (
        severity: string
    ) => {
        switch (severity) {
            case 'CRITICAL':
                return (
                    <ExclamationCircleOutlined
                        style={{
                            color: '#ff4d4f',
                        }}
                    />
                );

            case 'WARNING':
                return (
                    <WarningOutlined
                        style={{
                            color: '#faad14',
                        }}
                    />
                );

            case 'INFO':
                return (
                    <InfoCircleOutlined
                        style={{
                            color: '#1890ff',
                        }}
                    />
                );

            default:
                return null;
        }
    };

    const getSeverityColor = (
        severity: string
    ) => {
        switch (severity) {
            case 'CRITICAL':
                return 'red';

            case 'WARNING':
                return 'orange';

            case 'INFO':
                return 'blue';

            default:
                return 'default';
        }
    };

    const getStatusColor = (
        status: string
    ) => {
        switch (status) {
            case 'ACTIVE':
                return 'red';

            case 'ACKNOWLEDGED':
                return 'orange';

            case 'RESOLVED':
                return 'green';

            default:
                return 'default';
        }
    };

    const openDetail = (
        record: Alert
    ) => {
        setSelectedAlert(record);
        setDetailDrawer(true);
    };

    const columns = [
        {
            title: 'Severity',
            dataIndex: 'severity',
            key: 'severity',
            width: 110,

            render: (
                severity: string
            ) => (
                <Space>
                    {getSeverityIcon(
                        severity
                    )}

                    <Tag
                        color={getSeverityColor(
                            severity
                        )}
                    >
                        {severity}
                    </Tag>
                </Space>
            ),
        },

        {
            title: 'Alert Type',
            dataIndex: 'alert_type',
            key: 'alert_type',
            width: 140,

            render: (
                type: string
            ) => (
                <Tag>
                    {type}
                </Tag>
            ),
        },

        {
            title: 'Model',
            dataIndex: 'model_id',
            key: 'model_id',
            width: 175,

            render: (
                model_id: string
            ) => (
                <Space>
                    <RobotOutlined
                        style={{
                            color: '#722ed1',
                        }}
                    />

                    <Text
                        code
                        style={{
                            fontSize: 12,
                        }}
                    >
                        {model_id}
                    </Text>
                </Space>
            ),
        },

        {
            title: 'Title',
            dataIndex: 'title',
            key: 'title',

            render: (
                title: string,
                record: Alert
            ) => (
                <Text
                    strong
                    style={{
                        cursor: 'pointer',
                        color: '#1677ff',
                    }}
                    onClick={(e) => {
                        e.stopPropagation();
                        openDetail(record);
                    }}
                >
                    {title}
                </Text>
            ),
        },

        {
            title: 'Message',
            dataIndex: 'message',
            key: 'message',
            ellipsis: true,
        },

        {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',
            width: 130,

            render: (
                status: string
            ) => (
                <Tag
                    color={getStatusColor(
                        status
                    )}
                >
                    {status}
                </Tag>
            ),
        },

        {
            title: 'Created',
            dataIndex: 'created_at',
            key: 'created_at',
            width: 155,

            render: (
                date: string
            ) =>
                new Date(
                    date
                ).toLocaleString(),
        },

        {
            title: 'Actions',
            key: 'actions',
            width: 110,

            render: (
                _: unknown,
                record: Alert
            ) => (
                <Button
                    size="small"
                    onClick={(e) => {
                        e.stopPropagation();
                        openDetail(record);
                    }}
                >
                    View Details
                </Button>
            ),
        },
    ];

    const summary =
        alertsData?.summary || {
            active: 0,
            acknowledged: 0,
            resolved: 0,
            critical: 0,
        };

    const alerts =
        alertsData?.data || [];

    const criticalCount =
        summary.critical ??
        alerts.filter(
            (a: Alert) =>
                a.severity ===
                'CRITICAL'
        ).length;

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
                        Alerts
                    </Title>

                    <Text type="secondary">
                        Monitor anomaly detection,
                        model, data, and MLOps alerts
                    </Text>
                </div>
            </div>

            {/* Summary Cards */}
            <Row
                gutter={16}
                style={{
                    marginBottom: 24,
                }}
            >
                <Col span={6}>
                    <Card>
                        <Statistic
                            title="Active Alerts"
                            value={
                                summary.active
                            }
                            prefix={
                                <Badge
                                    status="error"
                                />
                            }
                            valueStyle={{
                                color:
                                    summary.active >
                                    0
                                        ? '#cf1322'
                                        : '#3f8600',
                            }}
                        />
                    </Card>
                </Col>

                <Col span={6}>
                    <Card>
                        <Statistic
                            title="Critical"
                            value={
                                criticalCount
                            }
                            prefix={
                                <ExclamationCircleOutlined
                                    style={{
                                        color: '#ff4d4f',
                                    }}
                                />
                            }
                            valueStyle={{
                                color:
                                    criticalCount >
                                    0
                                        ? '#cf1322'
                                        : '#3f8600',
                            }}
                        />
                    </Card>
                </Col>

                <Col span={6}>
                    <Card>
                        <Statistic
                            title="Acknowledged"
                            value={
                                summary.acknowledged
                            }
                            prefix={
                                <ClockCircleOutlined
                                    style={{
                                        color: '#faad14',
                                    }}
                                />
                            }
                        />
                    </Card>
                </Col>

                <Col span={6}>
                    <Card>
                        <Statistic
                            title="Resolved"
                            value={
                                summary.resolved
                            }
                            prefix={
                                <CheckCircleOutlined
                                    style={{
                                        color: '#52c41a',
                                    }}
                                />
                            }
                            valueStyle={{
                                color: '#3f8600',
                            }}
                        />
                    </Card>
                </Col>
            </Row>

            {/* Alerts Table */}
            <Card title="Anomaly Detection & MLOps Alerts">
                <Table
                    loading={isLoading}
                    dataSource={alerts}
                    columns={columns}
                    rowKey="id"
                    onRow={(
                        record: Alert
                    ) => ({
                        onClick: () =>
                            openDetail(
                                record
                            ),
                        style: {
                            cursor: 'pointer',
                        },
                    })}
                    pagination={{
                        pageSize: 10,

                        showTotal: (
                            total: number
                        ) =>
                            `Total ${total} alerts`,
                    }}
                    rowClassName={(
                        record: Alert
                    ) =>
                        record.status ===
                            'ACTIVE' &&
                        record.severity ===
                            'CRITICAL'
                            ? 'alert-critical'
                            : ''
                    }
                />
            </Card>

            {/* Alert Detail Drawer */}
            <Drawer
                title={
                    <Space>
                        {selectedAlert &&
                            getSeverityIcon(
                                selectedAlert.severity
                            )}

                        <span>
                            {
                                selectedAlert?.title
                            }
                        </span>
                    </Space>
                }
                open={detailDrawer}
                width={500}
                onClose={() => {
                    setDetailDrawer(
                        false
                    );
                    setSelectedAlert(
                        null
                    );
                }}
                extra={
                    selectedAlert && (
                        <Space>
                            {selectedAlert.status ===
                                'ACTIVE' &&
                                !isInformationalTrainingAlert(
                                    selectedAlert
                                ) && (
                                    <Button
                                        onClick={() =>
                                            setAcknowledgeModal(
                                                true
                                            )
                                        }
                                    >
                                        Acknowledge
                                    </Button>
                                )}

                            {selectedAlert.status ===
                                'ACKNOWLEDGED' &&
                                !isInformationalTrainingAlert(
                                    selectedAlert
                                ) && (
                                    <Button
                                        type="primary"
                                        onClick={() =>
                                            resolveMutation.mutate(
                                                {
                                                    alertId:
                                                        selectedAlert.id,
                                                }
                                            )
                                        }
                                        loading={
                                            resolveMutation.isPending
                                        }
                                    >
                                        Resolve
                                    </Button>
                                )}
                        </Space>
                    )
                }
            >
                {selectedAlert && (
                    <>
                        <Descriptions
                            column={1}
                            bordered
                            size="small"
                        >
                            <Descriptions.Item label="Model ID">
                                <Space>
                                    <RobotOutlined
                                        style={{
                                            color: '#722ed1',
                                        }}
                                    />

                                    <Text code>
                                        {
                                            selectedAlert.model_id
                                        }
                                    </Text>
                                </Space>
                            </Descriptions.Item>

                            <Descriptions.Item label="Alert Type">
                                <Tag>
                                    {
                                        selectedAlert.alert_type
                                    }
                                </Tag>
                            </Descriptions.Item>

                            <Descriptions.Item label="Severity">
                                <Tag
                                    color={getSeverityColor(
                                        selectedAlert.severity
                                    )}
                                >
                                    {
                                        selectedAlert.severity
                                    }
                                </Tag>
                            </Descriptions.Item>

                            <Descriptions.Item label="Status">
                                <Tag
                                    color={getStatusColor(
                                        selectedAlert.status
                                    )}
                                >
                                    {
                                        selectedAlert.status
                                    }
                                </Tag>
                            </Descriptions.Item>

                            <Descriptions.Item label="Created At">
                                {new Date(
                                    selectedAlert.created_at
                                ).toLocaleString()}
                            </Descriptions.Item>

                            {selectedAlert.acknowledged_at && (
                                <Descriptions.Item label="Acknowledged At">
                                    {new Date(
                                        selectedAlert.acknowledged_at
                                    ).toLocaleString()}
                                </Descriptions.Item>
                            )}

                            {selectedAlert.acknowledged_by && (
                                <Descriptions.Item label="Acknowledged By">
                                    <Text code>
                                        {
                                            selectedAlert.acknowledged_by
                                        }
                                    </Text>
                                </Descriptions.Item>
                            )}

                            {selectedAlert.resolved_at && (
                                <Descriptions.Item label="Resolved At">
                                    {new Date(
                                        selectedAlert.resolved_at
                                    ).toLocaleString()}
                                </Descriptions.Item>
                            )}
                        </Descriptions>

                        <Divider>
                            Alert Message
                        </Divider>

                        <Card
                            size="small"
                            style={{
                                background:
                                    '#fafafa',
                            }}
                        >
                            <Text>
                                {
                                    selectedAlert.message
                                }
                            </Text>
                        </Card>

                        {isInformationalTrainingAlert(
                            selectedAlert
                        ) && (
                            <>
                                <Divider />

                                <Text type="secondary">
                                    This is an
                                    informational
                                    training
                                    completion
                                    notification
                                    and does not
                                    require
                                    acknowledgement.
                                </Text>
                            </>
                        )}

                        {selectedAlert.details &&
                            Object.keys(
                                selectedAlert.details
                            ).length >
                                0 && (
                                <>
                                    <Divider>
                                        Technical
                                        Details
                                    </Divider>

                                    <Descriptions
                                        column={1}
                                        bordered
                                        size="small"
                                    >
                                        {Object.entries(
                                            selectedAlert.details
                                        ).map(
                                            (
                                                [
                                                    k,
                                                    v,
                                                ]
                                            ) => (
                                                <Descriptions.Item
                                                    key={
                                                        k
                                                    }
                                                    label={
                                                        k
                                                    }
                                                >
                                                    {typeof v ===
                                                    'number'
                                                        ? v.toFixed(
                                                              4
                                                          )
                                                        : String(
                                                              v
                                                          )}
                                                </Descriptions.Item>
                                            )
                                        )}
                                    </Descriptions>
                                </>
                            )}

                        {selectedAlert.resolution_note && (
                            <>
                                <Divider>
                                    Resolution Note
                                </Divider>

                                <Card
                                    size="small"
                                    style={{
                                        background:
                                            '#f6ffed',
                                    }}
                                >
                                    <Text>
                                        {
                                            selectedAlert.resolution_note
                                        }
                                    </Text>
                                </Card>
                            </>
                        )}
                    </>
                )}
            </Drawer>

            {/* Acknowledge Modal */}
            <Modal
                title="Acknowledge Alert"
                open={acknowledgeModal}
                onCancel={() => {
                    setAcknowledgeModal(
                        false
                    );
                    setResolutionNote(
                        ''
                    );
                }}
                onOk={() => {
                    if (
                        selectedAlert
                    ) {
                        acknowledgeMutation.mutate(
                            {
                                alertId:
                                    selectedAlert.id,
                                note: resolutionNote,
                            }
                        );
                    }
                }}
                confirmLoading={
                    acknowledgeMutation.isPending
                }
            >
                {selectedAlert && (
                    <>
                        <p>
                            <strong>
                                Model:
                            </strong>{' '}
                            <Text code>
                                {
                                    selectedAlert.model_id
                                }
                            </Text>
                        </p>

                        <p>
                            <strong>
                                Alert:
                            </strong>{' '}
                            {
                                selectedAlert.title
                            }
                        </p>

                        <p>
                            <strong>
                                Message:
                            </strong>{' '}
                            {
                                selectedAlert.message
                            }
                        </p>

                        <Input.TextArea
                            placeholder="Add a note (optional)"
                            rows={4}
                            value={
                                resolutionNote
                            }
                            onChange={(e) =>
                                setResolutionNote(
                                    e.target
                                        .value
                                )
                            }
                        />
                    </>
                )}
            </Modal>
        </div>
    );
}