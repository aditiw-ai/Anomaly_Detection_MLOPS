/**
 * Datasets Page
 * Upload, preview, download, and manage datasets used by the anomaly detection pipeline.
 */

import { useState } from 'react';
import {
    Alert,
    Card,
    Table,
    Button,
    Modal,
    Form,
    Input,
    Upload,
    message,
    Tag,
    Typography,
    Space,
    Tabs,
    Tooltip,
} from 'antd';
import {
    PlusOutlined,
    UploadOutlined,
    EyeOutlined,
    DeleteOutlined,
    DownloadOutlined,
} from '@ant-design/icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { datasetService } from '@/services/datasetService';
import { useAuth } from '@/contexts/AuthContext';

import type { UploadFile } from 'antd/es/upload/interface';

const { Title, Text, Paragraph } = Typography;

export function DataRegistry() {
    const { hasRole } = useAuth();
    const canWrite = hasRole(['ADMIN', 'DATA_ENGINEER']);

    const [uploadModalOpen, setUploadModalOpen] = useState(false);
    const [deleteModalOpen, setDeleteModalOpen] = useState(false);
    const [datasetToDelete, setDatasetToDelete] = useState<{
        id: string;
        name: string;
    } | null>(null);

    const [previewModalOpen, setPreviewModalOpen] = useState(false);
    const [previewData, setPreviewData] = useState<{
        columns: string[];
        rows: unknown[];
        total_rows: number;
    } | null>(null);

    const [previewLoading, setPreviewLoading] = useState(false);

    const [form] = Form.useForm();
    const [fileList, setFileList] = useState<UploadFile[]>([]);
    const queryClient = useQueryClient();

    // Fetch datasets
    const { data, isLoading, isError, error } = useQuery({
        queryKey: ['datasets', { type: 'raw' }],

        queryFn: async () => {
            const res = await datasetService.list(
                1,
                100,
                undefined,
                false,
                'raw'
            );

            return {
                ...res,
                data: res.data.filter(
                    (d) => !d.description?.startsWith('Merged from:')
                ),
            };
        },

        retry: 1,

        refetchInterval: (query) =>
            query.state.status === 'error' ? 15000 : 5000,
    });

    // Upload mutation
    const uploadMutation = useMutation({
        mutationFn: async (values: {
            name: string;
            description: string;
        }) => {
            if (
                fileList.length === 0 ||
                !fileList[0].originFileObj
            ) {
                throw new Error('Please select a file');
            }

            return datasetService.create(
                values.name,
                fileList[0].originFileObj,
                values.description
            );
        },

        onSuccess: () => {
            queryClient.invalidateQueries({
                queryKey: ['datasets'],
            });

            setUploadModalOpen(false);
            form.resetFields();
            setFileList([]);

            message.success('Dataset uploaded successfully');
        },

        onError: (error: any) => {
            message.error(
                error.response?.data?.detail ||
                    error.message ||
                    'Failed to upload dataset'
            );
        },
    });

    // Delete mutation
    const deleteMutation = useMutation({
        mutationFn: (id: string) => datasetService.delete(id),

        onSuccess: () => {
            queryClient.invalidateQueries({
                queryKey: ['datasets'],
            });

            message.success('Dataset deleted successfully');

            setDeleteModalOpen(false);
            setDatasetToDelete(null);
        },

        onError: () => {
            message.error('Failed to delete dataset');
        },
    });

    const handleDeleteClick = (id: string, name: string) => {
        setDatasetToDelete({ id, name });
        setDeleteModalOpen(true);
    };

    const handleConfirmDelete = () => {
        if (datasetToDelete) {
            deleteMutation.mutate(datasetToDelete.id);
        }
    };

    const [previewDataset, setPreviewDataset] = useState<{
        id: string;
        name: string;
        file_format: string;
        row_count: number;
        file_size_bytes: number;
        description?: string;
        schema?: {
            columns: {
                name: string;
                type: string;
            }[];
        };
    } | null>(null);

    const handlePreview = async (record: any) => {
        setPreviewDataset(record);
        setPreviewLoading(true);
        setPreviewModalOpen(true);

        try {
            const preview = await datasetService.preview(
                record.id,
                10
            );

            setPreviewData(preview);
        } catch (error) {
            message.error('Failed to load dataset preview');
            setPreviewModalOpen(false);
        } finally {
            setPreviewLoading(false);
        }
    };

    const handleDownload = async () => {
        if (!previewDataset) {
            return;
        }

        try {
            message.loading({
                content: 'Preparing dataset download...',
                key: 'download',
            });

            const { download_url } =
                await datasetService.getDownloadUrl(
                    previewDataset.id
                );

            // Try File System Access API
            if ((window as any).showSaveFilePicker) {
                try {
                    // @ts-ignore
                    const handle =
                        await (window as any).showSaveFilePicker({
                            suggestedName: `${previewDataset.name}.${previewDataset.file_format}`,

                            types: [
                                {
                                    description: 'Dataset File',

                                    accept: {
                                        'text/csv': ['.csv'],
                                        'application/json': ['.json'],
                                        'application/octet-stream': [
                                            '.parquet',
                                        ],
                                    },
                                },
                            ],
                        });

                    const writable =
                        await handle.createWritable();

                    const response =
                        await fetch(download_url);

                    await response.body?.pipeTo(writable);

                    message.success({
                        content: 'Dataset saved!',
                        key: 'download',
                    });

                    return;
                } catch (err: any) {
                    if (err.name === 'AbortError') {
                        message.destroy('download');
                        return;
                    }

                    console.warn(
                        'File System Access API failed, falling back...',
                        err
                    );
                }
            }

            // Fallback
            window.open(download_url, '_blank');

            message.success({
                content: 'Download started!',
                key: 'download',
            });
        } catch (error) {
            console.error(error);

            message.error({
                content: 'Failed to download dataset',
                key: 'download',
            });
        }
    };

    const columns = [
        {
            title: 'Name',
            dataIndex: 'name',
            key: 'name',

            render: (name: string) => (
                <Text strong>{name}</Text>
            ),
        },

        {
            title: 'Version',
            dataIndex: 'version',
            key: 'version',

            render: (version: string) => (
                <Tag>{version}</Tag>
            ),
        },

        {
            title: 'Rows',
            dataIndex: 'row_count',
            key: 'row_count',

            render: (count: number) =>
                count?.toLocaleString() || '-',
        },

        {
            title: 'Columns',
            dataIndex: 'column_count',
            key: 'column_count',
        },

        {
            title: 'Status',
            dataIndex: 'status',
            key: 'status',

            render: (status: string) => {
                const colors: Record<string, string> = {
                    ACTIVE: 'green',
                    PROCESSING: 'blue',
                    ARCHIVED: 'gray',
                };

                return (
                    <Tag color={colors[status] || 'default'}>
                        {status}
                    </Tag>
                );
            },
        },

        {
            title: 'Created',
            dataIndex: 'created_at',
            key: 'created_at',

            render: (date: string) =>
                new Date(date).toLocaleDateString(),
        },

        {
            title: 'File Type',
            dataIndex: 'file_format',
            key: 'file_format',

            render: (format: string) => (
                <Tag
                    color={
                        format === 'csv'
                            ? 'blue'
                            : format === 'parquet'
                            ? 'purple'
                            : 'orange'
                    }
                >
                    {format?.toUpperCase() || '-'}
                </Tag>
            ),
        },

        {
            title: 'Actions',
            key: 'actions',

            render: (
                _: unknown,
                record: {
                    id: string;
                    name: string;
                }
            ) => (
                <Space>
                    <Button
                        size="small"
                        icon={<EyeOutlined />}
                        onClick={() =>
                            handlePreview(record)
                        }
                        title="Preview dataset"
                    />

                    <Tooltip
                        title={
                            !canWrite
                                ? 'Your role does not have permission to delete datasets.'
                                : 'Delete dataset'
                        }
                    >
                        <Button
                            size="small"
                            danger
                            icon={<DeleteOutlined />}
                            onClick={() =>
                                handleDeleteClick(
                                    record.id,
                                    record.name
                                )
                            }
                            loading={
                                deleteMutation.isPending &&
                                datasetToDelete?.id ===
                                    record.id
                            }
                            disabled={!canWrite}
                        />
                    </Tooltip>
                </Space>
            ),
        },
    ];

    return (
        <div className="fade-in">
            {/* Header */}
            <div className="page-header">
                <div>
                    <Title
                        level={2}
                        style={{ margin: 0 }}
                    >
                        Datasets
                    </Title>

                    <Text type="secondary">
                        Manage datasets used by the anomaly detection
                        and MLOps pipeline
                    </Text>
                </div>

                <Tooltip
                    title={
                        !canWrite
                            ? 'Your role does not have permission to upload datasets.'
                            : ''
                    }
                >
                    <Button
                        type="primary"
                        icon={<PlusOutlined />}
                        onClick={() =>
                            setUploadModalOpen(true)
                        }
                        disabled={!canWrite}
                    >
                        Upload Dataset
                    </Button>
                </Tooltip>
            </div>

            <Card>
                {isError && (
                    <Alert
                        type="error"
                        showIcon
                        message="Unable to load datasets"
                        description={
                            (error as any)?.response?.data
                                ?.detail ||
                            (error as Error)?.message ||
                            'Failed to load datasets'
                        }
                        style={{ marginBottom: 16 }}
                    />
                )}

                <Table
                    dataSource={data?.data || []}
                    columns={columns}
                    loading={isLoading}
                    rowKey="id"
                    pagination={{
                        pageSize: 10,
                        showSizeChanger: true,
                        showTotal: (total) =>
                            `Total ${total} datasets`,
                    }}
                />
            </Card>

            {/* Upload Modal */}
            <Modal
                title="Upload Dataset"
                open={uploadModalOpen}
                onCancel={() => {
                    setUploadModalOpen(false);
                    form.resetFields();
                    setFileList([]);
                }}
                onOk={() => form.submit()}
                confirmLoading={uploadMutation.isPending}
            >
                <Form
                    form={form}
                    layout="vertical"
                    onFinish={(values) =>
                        uploadMutation.mutate(values)
                    }
                >
                    <Form.Item
                        name="name"
                        label="Dataset Name"
                        rules={[
                            {
                                required: true,
                                message:
                                    'Please enter a dataset name',
                            },
                        ]}
                    >
                        <Input placeholder="e.g., anomaly_train_2026" />
                    </Form.Item>

                    <Form.Item
                        name="description"
                        label="Description"
                    >
                        <Input.TextArea
                            placeholder="Describe the dataset and its purpose"
                            rows={3}
                        />
                    </Form.Item>

                    <Form.Item
                        label="File"
                        required
                        rules={[
                            {
                                required: true,
                                message:
                                    'Please upload a dataset file',
                            },
                        ]}
                    >
                        <Upload
                            beforeUpload={() => false}
                            fileList={fileList}
                            onChange={({ fileList }) => {
                                setFileList(fileList);

                                if (
                                    fileList.length > 0 &&
                                    fileList[0].name
                                ) {
                                    const fileName =
                                        fileList[0].name;

                                    const nameWithoutExt =
                                        fileName.substring(
                                            0,
                                            fileName.lastIndexOf(
                                                '.'
                                            )
                                        ) || fileName;

                                    form.setFieldValue(
                                        'name',
                                        nameWithoutExt
                                    );
                                }
                            }}
                            accept=".csv,.parquet,.json"
                            maxCount={1}
                        >
                            <Button
                                icon={<UploadOutlined />}
                            >
                                Select File
                            </Button>
                        </Upload>

                        <Text
                            type="secondary"
                            style={{
                                display: 'block',
                                marginTop: 8,
                            }}
                        >
                            Supported formats: CSV, Parquet, JSON
                        </Text>
                    </Form.Item>
                </Form>
            </Modal>

            {/* Delete Confirmation Modal */}
            <Modal
                title="Delete Dataset"
                open={deleteModalOpen}
                onCancel={() => {
                    setDeleteModalOpen(false);
                    setDatasetToDelete(null);
                }}
                onOk={handleConfirmDelete}
                okText="Delete"
                okType="danger"
                confirmLoading={deleteMutation.isPending}
            >
                <p>
                    Are you sure you want to delete "
                    {datasetToDelete?.name}"?
                </p>

                <p style={{ color: '#ff4d4f' }}>
                    This action cannot be undone.
                </p>
            </Modal>

            {/* Preview Modal */}
            <Modal
                title={`Preview: ${
                    previewDataset?.name || ''
                }`}
                open={previewModalOpen}
                onCancel={() => {
                    setPreviewModalOpen(false);
                    setPreviewData(null);
                    setPreviewDataset(null);
                }}
                footer={[
                    <Button
                        key="download"
                        type="primary"
                        icon={<DownloadOutlined />}
                        onClick={handleDownload}
                        disabled={!previewDataset}
                    >
                        Download
                    </Button>,

                    <Button
                        key="close"
                        onClick={() => {
                            setPreviewModalOpen(false);
                            setPreviewData(null);
                            setPreviewDataset(null);
                        }}
                    >
                        Close
                    </Button>,
                ]}
                width={800}
            >
                {previewLoading ? (
                    <div
                        style={{
                            textAlign: 'center',
                            padding: '40px',
                        }}
                    >
                        Loading dataset preview...
                    </div>
                ) : previewDataset ? (
                    <Tabs
                        defaultActiveKey="preview"
                        items={[
                            {
                                key: 'preview',
                                label: 'Data Preview',

                                children: (
                                    <>
                                        {previewData ? (
                                            <Table
                                                dataSource={previewData.rows.map(
                                                    (
                                                        row: any,
                                                        idx: number
                                                    ) => ({
                                                        key: idx,
                                                        ...(row as object),
                                                    })
                                                )}
                                                columns={previewData.columns.map(
                                                    (
                                                        col: string
                                                    ) => ({
                                                        title: col,
                                                        dataIndex:
                                                            col,
                                                        key: col,
                                                        ellipsis:
                                                            true,
                                                    })
                                                )}
                                                pagination={false}
                                                scroll={{ x: true }}
                                                size="small"
                                            />
                                        ) : (
                                            <div>
                                                No preview data
                                                available
                                            </div>
                                        )}

                                        {previewData && (
                                            <Text
                                                type="secondary"
                                                style={{
                                                    display: 'block',
                                                    marginTop: 16,
                                                }}
                                            >
                                                Showing{' '}
                                                {
                                                    previewData
                                                        .rows
                                                        .length
                                                }{' '}
                                                of{' '}
                                                {
                                                    previewData.total_rows
                                                }{' '}
                                                rows
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
                                        <div
                                            style={{
                                                marginBottom: 16,
                                            }}
                                        >
                                            <Tag color="blue">
                                                {previewDataset.file_format?.toUpperCase()}
                                            </Tag>

                                            <Tag color="cyan">
                                                {previewDataset.row_count?.toLocaleString()}{' '}
                                                rows
                                            </Tag>

                                            <Tag color="purple">
                                                {(previewDataset as any)
                                                    .column_count
                                                    ?.toLocaleString() ||
                                                    previewDataset
                                                        .schema
                                                        ?.columns
                                                        ?.length ||
                                                    0}{' '}
                                                columns
                                            </Tag>

                                            <Tag>
                                                {(
                                                    previewDataset.file_size_bytes
                                                        ? previewDataset.file_size_bytes /
                                                          1024
                                                        : 0
                                                ).toFixed(1)}{' '}
                                                KB
                                            </Tag>
                                        </div>

                                        {previewDataset.schema && (
                                            <Card
                                                type="inner"
                                                title="Schema"
                                                size="small"
                                            >
                                                <div
                                                    style={{
                                                        maxHeight: 300,
                                                        overflow:
                                                            'auto',
                                                    }}
                                                >
                                                    {previewDataset.schema.columns.map(
                                                        (
                                                            col
                                                        ) => (
                                                            <div
                                                                key={
                                                                    col.name
                                                                }
                                                                style={{
                                                                    display:
                                                                        'flex',
                                                                    justifyContent:
                                                                        'space-between',
                                                                    padding:
                                                                        '8px 0',
                                                                    borderBottom:
                                                                        '1px solid #f0f0f0',
                                                                }}
                                                            >
                                                                <Text strong>
                                                                    {
                                                                        col.name
                                                                    }
                                                                </Text>

                                                                <Tag>
                                                                    {
                                                                        col.type
                                                                    }
                                                                </Tag>
                                                            </div>
                                                        )
                                                    )}
                                                </div>
                                            </Card>
                                        )}

                                        {previewDataset.description && (
                                            <div
                                                style={{
                                                    marginTop: 16,
                                                }}
                                            >
                                                <Text strong>
                                                    Description:
                                                </Text>

                                                <Paragraph>
                                                    {
                                                        previewDataset.description
                                                    }
                                                </Paragraph>
                                            </div>
                                        )}
                                    </div>
                                ),
                            },
                        ]}
                    />
                ) : (
                    <div>No dataset available</div>
                )}
            </Modal>
        </div>
    );
}