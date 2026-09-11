/**
 * Inference Page
 * Real-time anomaly detection with model selection, single + batch prediction.
 * Form fields are dynamically generated from the loaded model's input features.
 *
 * Note:
 * The backend currently exposes some legacy wire-contract fields such as
 * `fraud_score` and `transaction_id`. Those are intentionally kept in the
 * service layer/API contract, while this UI presents them as anomaly data.
 */

import { useState } from 'react';
import {
    Card,
    Row,
    Col,
    Typography,
    Form,
    Input,
    Button,
    Statistic,
    Tag,
    Progress,
    Divider,
    Select,
    Table,
    message,
    Spin,
    Alert,
    Upload,
    Tooltip,
} from 'antd';
import {
    ThunderboltOutlined,
    WarningOutlined,
    ExclamationCircleOutlined,
    UploadOutlined,
    CloudServerOutlined,
    CheckCircleOutlined,
} from '@ant-design/icons';
import { useMutation, useQuery } from '@tanstack/react-query';
import {
    inferenceService,
    PredictionResponse,
    BatchPredictionResult,
    InferenceModel,
} from '@/services/inferenceService';
import { useAuth } from '@/contexts/AuthContext';

const { Title, Text } = Typography;
const { TextArea } = Input;

export function Inference() {
    const { hasRole } = useAuth();
    const canInfer = hasRole(['ADMIN', 'ML_ENGINEER', 'DEPLOYER']);

    const [singleResult, setSingleResult] =
        useState<PredictionResponse | null>(null);
    const [batchResults, setBatchResults] =
        useState<BatchPredictionResult[] | null>(null);
    const [batchMeta, setBatchMeta] = useState<any>(null);
    const [selectedModelId, setSelectedModelId] = useState<string | null>(
        null
    );
    const [modelLoaded, setModelLoaded] = useState(false);
    const [modelLoadError, setModelLoadError] = useState<string | null>(null);
    const [inputFeatures, setInputFeatures] = useState<string[]>([]);
    const [batchInput, setBatchInput] = useState('');
    const [selectedRiskFilter, setSelectedRiskFilter] = useState<string | null>(
        null
    );
    const [form] = Form.useForm();

    const handleRiskFilter = (riskLevel: string) => {
        setSelectedRiskFilter(
            selectedRiskFilter === riskLevel ? null : riskLevel
        );
    };

    const filteredBatchResults = selectedRiskFilter
        ? batchResults?.filter(
              (r) => r.risk_level === selectedRiskFilter
          )
        : batchResults;

    // Fetch available models
    const { data: modelsData, isLoading: modelsLoading } = useQuery({
        queryKey: ['inference-models'],
        queryFn: inferenceService.listModels,
    });

    const models = modelsData?.data || [];

    // Load model
    const loadModelMutation = useMutation({
        mutationFn: (modelId: string) => inferenceService.loadModel(modelId),

        onSuccess: (data) => {
            setModelLoaded(true);
            setModelLoadError(null);
            setSingleResult(null);
            setBatchResults(null);
            setBatchMeta(null);

            const features = data?.data?.input_features || [];
            setInputFeatures(features);
            form.resetFields();

            message.success(
                `Model loaded (${data?.data?.inference_engine || 'pickle'} engine)`
            );
        },

        onError: (err: any) => {
            const detail =
                err?.response?.data?.detail || 'Failed to load model';

            setModelLoaded(true);
            setModelLoadError(detail);
            setInputFeatures([]);

            message.warning(
                `Model load issue: ${detail}. Batch inference may still work if the server has a model loaded.`
            );
        },
    });

    // Single prediction
    const predictMutation = useMutation({
        mutationFn: inferenceService.predict,

        onSuccess: (data) => {
            setSingleResult(data);
        },

        onError: (err: any) => {
            message.error(
                err?.response?.data?.detail || 'Inference failed'
            );
        },
    });

    // Batch prediction
    const batchMutation = useMutation({
        mutationFn: inferenceService.predictBatch,

        onSuccess: (data) => {
            setBatchResults(data.data);
            setBatchMeta(data.meta);
        },

        onError: (err: any) => {
            message.error(
                err?.response?.data?.detail || 'Batch inference failed'
            );
        },
    });

    const handleModelSelect = (modelId: string) => {
        setSelectedModelId(modelId);
        setModelLoaded(false);
        setModelLoadError(null);
        setInputFeatures([]);
        setSingleResult(null);
        setBatchResults(null);
        setBatchMeta(null);

        loadModelMutation.mutate(modelId);
    };

    const handlePredict = (values: any) => {
        const features: Record<string, any> = {};

        for (const [key, val] of Object.entries(values)) {
            if (val === undefined || val === null || val === '') {
                continue;
            }

            const num = Number(val);
            features[key] = isNaN(num) ? val : num;
        }

        predictMutation.mutate({ features });
    };

    const handleBatchPredict = () => {
        try {
            const parsed = JSON.parse(batchInput);
            const records = Array.isArray(parsed) ? parsed : [parsed];

            batchMutation.mutate(records);
        } catch {
            message.error(
                'Invalid JSON. Provide an array of anomaly-data records.'
            );
        }
    };

    const getRiskColor = (riskLevel: string) => {
        const map: Record<string, string> = {
            CRITICAL: '#ff4d4f',
            HIGH: '#fa8c16',
            MEDIUM: '#faad14',
            LOW: '#52c41a',
        };

        return map[riskLevel] || '#8c8c8c';
    };

    const getRiskTag = (riskLevel: string) => {
        const colorMap: Record<string, string> = {
            CRITICAL: 'red',
            HIGH: 'orange',
            MEDIUM: 'gold',
            LOW: 'green',
        };

        return (
            <Tag color={colorMap[riskLevel] || 'default'}>
                {riskLevel}
            </Tag>
        );
    };

    const getPredictionLabel = (prediction: number) =>
        prediction === 1 ? 'ANOMALY' : 'NORMAL';

    const getPredictionColor = (prediction: number) =>
        prediction === 1 ? 'red' : 'green';

    const batchColumns = [
        {
            title: '#',
            dataIndex: 'index',
            key: 'index',
            width: 50,
        },
        {
            title: 'Stage 1 Result',
            dataIndex: 'prediction',
            key: 'prediction',
            render: (v: number) => (
                <Tag color={getPredictionColor(v)}>
                    {getPredictionLabel(v)}
                </Tag>
            ),
        },
        {
            title: 'Anomaly Score',
            dataIndex: 'fraud_score',
            key: 'fraud_score',
            render: (v: number) =>
                v != null ? `${(v * 100).toFixed(1)}%` : '-',
        },
        {
            title: 'Confidence',
            dataIndex: 'confidence',
            key: 'confidence',
            render: (v: number) =>
                v != null ? `${(v * 100).toFixed(1)}%` : '-',
        },
        {
            title: 'Risk',
            dataIndex: 'risk_level',
            key: 'risk_level',
            render: (v: string) => getRiskTag(v),
        },
    ];

    const formatLabel = (name: string) =>
        name
            .replace(/_/g, ' ')
            .replace(/\b\w/g, (c) => c.toUpperCase());

    return (
        <div className="fade-in">
            {/* Header */}
            <div
                className="page-header"
                style={{ marginBottom: 24 }}
            >
                <div style={{ flex: 1 }}>
                    <Title level={2} style={{ margin: 0 }}>
                        Anomaly Inference
                    </Title>

                    <Text type="secondary">
                        Run real-time anomaly detection using a registered
                        model
                    </Text>
                </div>

                <div
                    style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 12,
                    }}
                >
                    <CloudServerOutlined style={{ fontSize: 18 }} />

                    <Select
                        showSearch
                        filterOption={(input, option) =>
                            (
                                option?.searchName?.toString().toLowerCase() ??
                                ''
                            ).includes(input.toLowerCase())
                        }
                        placeholder="Select a model"
                        style={{ width: 320 }}
                        loading={modelsLoading}
                        value={selectedModelId}
                        onChange={handleModelSelect}
                        options={models.map((m: InferenceModel) => ({
                            value: m.model_id,
                            searchName: m.name,
                            label: (
                                <span>
                                    {m.name}

                                    <Tag
                                        color={
                                            m.status === 'PRODUCTION'
                                                ? 'green'
                                                : m.status === 'STAGING'
                                                ? 'blue'
                                                : 'default'
                                        }
                                        style={{
                                            marginLeft: 8,
                                            fontSize: 10,
                                        }}
                                    >
                                        {m.status}
                                    </Tag>
                                </span>
                            ),
                        }))}
                    />

                    {modelLoaded && (
                        <Tag
                            icon={<CheckCircleOutlined />}
                            color="success"
                        >
                            Loaded
                        </Tag>
                    )}

                    {loadModelMutation.isPending && (
                        <Spin size="small" />
                    )}
                </div>
            </div>

            {!modelLoaded && !loadModelMutation.isPending && (
                <Alert
                    message="Select a model to begin inference"
                    description="Choose a trained anomaly-detection model from the dropdown above."
                    type="info"
                    showIcon
                    style={{ marginBottom: 24 }}
                />
            )}

            <Row gutter={24}>
                {/* SINGLE INFERENCE */}
                <Col span={12}>
                    <Card
                        title={
                            <span>
                                Single Inference

                                {inputFeatures.length > 0 && (
                                    <Tag style={{ marginLeft: 8 }}>
                                        {inputFeatures.length} features
                                    </Tag>
                                )}
                            </span>
                        }
                        extra={<ThunderboltOutlined />}
                        style={{ marginBottom: 24 }}
                    >
                        {modelLoaded && inputFeatures.length > 0 ? (
                            <Form
                                form={form}
                                layout="vertical"
                                onFinish={handlePredict}
                                style={{
                                    maxHeight: 400,
                                    overflowY: 'auto',
                                    paddingRight: 8,
                                }}
                            >
                                {inputFeatures.map((featureName) => (
                                    <Form.Item
                                        key={featureName}
                                        name={featureName}
                                        label={formatLabel(featureName)}
                                    >
                                        <Input placeholder={featureName} />
                                    </Form.Item>
                                ))}

                                <Tooltip
                                    title={
                                        !canInfer
                                            ? 'Your role does not have permission to run inference.'
                                            : ''
                                    }
                                >
                                    <Button
                                        type="primary"
                                        htmlType="submit"
                                        loading={predictMutation.isPending}
                                        icon={<ThunderboltOutlined />}
                                        size="large"
                                        style={{ width: '100%' }}
                                        disabled={!canInfer}
                                    >
                                        Run Inference
                                    </Button>
                                </Tooltip>
                            </Form>
                        ) : modelLoaded && modelLoadError ? (
                            <Alert
                                message="Model Load Warning"
                                description={`${modelLoadError}. Use batch inference with JSON input if available.`}
                                type="warning"
                                showIcon
                            />
                        ) : modelLoaded ? (
                            <Alert
                                message="Input features not available"
                                description="This model does not expose dynamic input features. Use batch inference with JSON input instead."
                                type="warning"
                                showIcon
                            />
                        ) : (
                            <div
                                style={{
                                    textAlign: 'center',
                                    padding: 32,
                                    color: '#bfbfbf',
                                }}
                            >
                                <ThunderboltOutlined
                                    style={{ fontSize: 36 }}
                                />

                                <Title
                                    level={5}
                                    type="secondary"
                                    style={{ marginTop: 12 }}
                                >
                                    Load a model to see input fields
                                </Title>
                            </div>
                        )}
                    </Card>

                    {/* SINGLE RESULT */}
                    {singleResult && (
                        <Card>
                            <Alert
                                message={
                                    singleResult.prediction === 1
                                        ? 'Stage 1 Result: Anomaly Detected'
                                        : 'Stage 1 Result: Normal'
                                }
                                type={
                                    singleResult.prediction === 1
                                        ? 'error'
                                        : 'success'
                                }
                                showIcon
                                icon={
                                    singleResult.prediction === 1 ? (
                                        <WarningOutlined />
                                    ) : (
                                        <CheckCircleOutlined />
                                    )
                                }
                                style={{ marginBottom: 24 }}
                            />

                            <Row
                                gutter={16}
                                style={{ marginBottom: 24 }}
                            >
                                <Col span={8}>
                                    <Card size="small">
                                        <Statistic
                                            title="Stage 1"
                                            value={
                                                singleResult.prediction === 1
                                                    ? 'ANOMALY'
                                                    : 'NORMAL'
                                            }
                                            valueStyle={{
                                                color:
                                                    singleResult.prediction ===
                                                    1
                                                        ? '#ff4d4f'
                                                        : '#52c41a',
                                                fontSize: 18,
                                            }}
                                        />
                                    </Card>
                                </Col>

                                <Col span={8}>
                                    <Card size="small">
                                        <Statistic
                                            title="Anomaly Score"
                                            value={
                                                singleResult.fraud_score != null
                                                    ? singleResult.fraud_score *
                                                      100
                                                    : '-'
                                            }
                                            suffix={
                                                singleResult.fraud_score != null
                                                    ? '%'
                                                    : ''
                                            }
                                            precision={1}
                                        />
                                    </Card>
                                </Col>

                                <Col span={8}>
                                    <Card size="small">
                                        <Statistic
                                            title="Response Time"
                                            value={
                                                singleResult.response_time_ms
                                            }
                                            suffix="ms"
                                            precision={2}
                                        />
                                    </Card>
                                </Col>
                            </Row>

                            <Divider />

                            <div>
                                <Text strong>Stage 1 Confidence</Text>

                                <Progress
                                    percent={
                                        singleResult.confidence != null
                                            ? singleResult.confidence * 100
                                            : 0
                                    }
                                    format={(p) =>
                                        `${p?.toFixed(1)}%`
                                    }
                                />
                            </div>

                            {/* Stage 2 is displayed only when the backend
                                actually returns anomaly-type information. */}
                            {(singleResult as any).anomaly_type && (
                                <>
                                    <Divider />

                                    <div>
                                        <Text strong>
                                            Stage 2 — Anomaly Type
                                        </Text>

                                        <div style={{ marginTop: 12 }}>
                                            <Tag
                                                color="orange"
                                                style={{
                                                    fontSize: 15,
                                                    padding: '6px 12px',
                                                }}
                                            >
                                                {
                                                    (singleResult as any)
                                                        .anomaly_type
                                                }
                                            </Tag>
                                        </div>
                                    </div>
                                </>
                            )}
                        </Card>
                    )}
                </Col>

                {/* BATCH INFERENCE */}
                <Col span={12}>
                    <Card
                        title="Batch Inference"
                        extra={<UploadOutlined />}
                        style={{ marginBottom: 24 }}
                    >
                        <Upload.Dragger
                            name="file"
                            multiple={false}
                            accept=".json,.csv"
                            showUploadList={false}
                            beforeUpload={(file) => {
                                const reader = new FileReader();

                                reader.onload = (e) => {
                                    try {
                                        const text =
                                            e.target?.result as string;

                                        if (file.name.endsWith('.json')) {
                                            const json = JSON.parse(text);

                                            setBatchInput(
                                                JSON.stringify(
                                                    json,
                                                    null,
                                                    2
                                                )
                                            );

                                            message.success(
                                                'JSON loaded successfully'
                                            );
                                        } else if (
                                            file.name.endsWith('.csv')
                                        ) {
                                            const parseCSVLine = (
                                                line: string
                                            ): string[] => {
                                                const values: string[] = [];
                                                let current = '';
                                                let inQuote = false;

                                                for (
                                                    let i = 0;
                                                    i < line.length;
                                                    i++
                                                ) {
                                                    const char = line[i];
                                                    const nextChar =
                                                        i + 1 < line.length
                                                            ? line[i + 1]
                                                            : null;

                                                    if (
                                                        char === '"' &&
                                                        nextChar === '"'
                                                    ) {
                                                        current += '"';
                                                        i++;
                                                    } else if (
                                                        char === '"'
                                                    ) {
                                                        inQuote = !inQuote;
                                                    } else if (
                                                        char === ',' &&
                                                        !inQuote
                                                    ) {
                                                        values.push(
                                                            current.trim()
                                                        );
                                                        current = '';
                                                    } else {
                                                        current += char;
                                                    }
                                                }

                                                values.push(current.trim());
                                                return values;
                                            };

                                            const lines = text
                                                .split(/\r?\n/)
                                                .filter(
                                                    (line) => line.trim()
                                                );

                                            if (lines.length < 2) {
                                                throw new Error(
                                                    'CSV must have a header and at least one row'
                                                );
                                            }

                                            const headers =
                                                parseCSVLine(lines[0]);

                                            if (inputFeatures.length > 0) {
                                                const missing =
                                                    inputFeatures.filter(
                                                        (f) =>
                                                            !headers.includes(
                                                                f
                                                            )
                                                    );

                                                if (missing.length > 0) {
                                                    message.warning(
                                                        `CSV is missing ${missing.length} columns: ${missing
                                                            .slice(0, 3)
                                                            .join(', ')}...`
                                                    );
                                                }
                                            }

                                            const data = lines
                                                .slice(1)
                                                .map((line) => {
                                                    const values =
                                                        parseCSVLine(line);

                                                    const obj: Record<
                                                        string,
                                                        any
                                                    > = {};

                                                    headers.forEach(
                                                        (h, i) => {
                                                            const val =
                                                                values[i];

                                                            if (
                                                                val !==
                                                                    undefined &&
                                                                val !== ''
                                                            ) {
                                                                const num =
                                                                    Number(
                                                                        val
                                                                    );

                                                                obj[h] =
                                                                    isNaN(num)
                                                                        ? val
                                                                        : num;
                                                            }
                                                        }
                                                    );

                                                    return obj;
                                                });

                                            const completeData =
                                                data.filter((row) => {
                                                    const nonNullFields =
                                                        Object.values(
                                                            row
                                                        ).filter(
                                                            (v) =>
                                                                v !==
                                                                    undefined &&
                                                                v !== null &&
                                                                v !== ''
                                                        ).length;

                                                    return (
                                                        nonNullFields /
                                                            headers.length >
                                                        0.5
                                                    );
                                                });

                                            const removedCount =
                                                data.length -
                                                completeData.length;

                                            if (removedCount > 0) {
                                                message.warning(
                                                    `Removed ${removedCount} incomplete rows.`
                                                );
                                            }

                                            setBatchInput(
                                                JSON.stringify(
                                                    completeData,
                                                    null,
                                                    2
                                                )
                                            );

                                            message.success(
                                                `CSV loaded (${completeData.length} valid rows)`
                                            );
                                        }
                                    } catch (err) {
                                        message.error(
                                            'Failed to parse file: ' +
                                                String(err)
                                        );
                                    }
                                };

                                reader.readAsText(file);
                                return false;
                            }}
                            style={{ marginBottom: 16 }}
                        >
                            <p className="ant-upload-drag-icon">
                                <UploadOutlined />
                            </p>

                            <p className="ant-upload-text">
                                Upload anomaly-data
                            </p>

                            <p className="ant-upload-hint">
                                Upload a CSV or JSON file for batch anomaly
                                inference.
                            </p>
                        </Upload.Dragger>

                        <Text
                            type="secondary"
                            style={{
                                display: 'block',
                                marginBottom: 12,
                            }}
                        >
                            Or paste a JSON array manually:
                        </Text>

                        <TextArea
                            rows={6}
                            placeholder={
                                inputFeatures.length > 0
                                    ? `[\n  {${inputFeatures
                                          .slice(0, 3)
                                          .map(
                                              (f) => `"${f}": ""`
                                          )
                                          .join(', ')}},\n  ...\n]`
                                    : '[\n  {"feature_1": "value", "feature_2": "value"},\n  ...\n]'
                            }
                            value={batchInput}
                            onChange={(e) =>
                                setBatchInput(e.target.value)
                            }
                            style={{
                                fontFamily: 'monospace',
                                fontSize: 13,
                            }}
                        />

                        <Tooltip
                            title={
                                !canInfer
                                    ? 'Your role does not have permission to run inference.'
                                    : ''
                            }
                        >
                            <Button
                                type="primary"
                                onClick={handleBatchPredict}
                                loading={batchMutation.isPending}
                                icon={<ThunderboltOutlined />}
                                size="large"
                                style={{
                                    width: '100%',
                                    marginTop: 16,
                                }}
                                disabled={
                                    !selectedModelId ||
                                    !batchInput.trim() ||
                                    !canInfer
                                }
                            >
                                Run Batch Inference
                            </Button>
                        </Tooltip>
                    </Card>

                    {/* BATCH RESULTS */}
                    {batchResults ? (
                        <Card>
                            {batchMeta && (
                                <>
                                    <Row
                                        gutter={16}
                                        style={{
                                            marginBottom: 20,
                                        }}
                                    >
                                        <Col span={8}>
                                            <Statistic
                                                title="Records"
                                                value={
                                                    batchMeta.total_transactions ??
                                                    batchResults.length
                                                }
                                            />
                                        </Col>

                                        <Col span={8}>
                                            <Statistic
                                                title="Anomalies"
                                                value={
                                                    batchMeta.fraud_count ?? '-'
                                                }
                                                valueStyle={{
                                                    color: '#cf1322',
                                                }}
                                            />
                                        </Col>

                                        <Col span={8}>
                                            <Statistic
                                                title="Total Time"
                                                value={
                                                    batchMeta.total_time_ms ??
                                                    '-'
                                                }
                                                suffix={
                                                    batchMeta.total_time_ms !=
                                                    null
                                                        ? 'ms'
                                                        : ''
                                                }
                                                precision={0}
                                            />
                                        </Col>
                                    </Row>

                                    <div
                                        style={{
                                            marginBottom: 20,
                                        }}
                                    >
                                        <Text
                                            type="secondary"
                                            style={{
                                                fontSize: 12,
                                            }}
                                        >
                                            Anomaly Rate:{' '}
                                            {batchMeta.total_transactions
                                                ? (
                                                      ((batchMeta.fraud_count ||
                                                          0) /
                                                          batchMeta.total_transactions) *
                                                      100
                                                  ).toFixed(1)
                                                : '0.0'}
                                            %
                                        </Text>

                                        <div
                                            style={{
                                                marginTop: 8,
                                            }}
                                        >
                                            {Object.entries(
                                                batchMeta.risk_summary || {}
                                            ).map(
                                                ([
                                                    level,
                                                    count,
                                                ]: [string, any]) => {
                                                    if (count === 0) {
                                                        return null;
                                                    }

                                                    const isSelected =
                                                        selectedRiskFilter ===
                                                        level;

                                                    const isDimmed =
                                                        selectedRiskFilter &&
                                                        !isSelected;

                                                    return (
                                                        <Tag
                                                            key={level}
                                                            color={getRiskColor(
                                                                level
                                                            )}
                                                            style={{
                                                                marginRight: 6,
                                                                cursor: 'pointer',
                                                                opacity:
                                                                    isDimmed
                                                                        ? 0.3
                                                                        : 1,
                                                                border:
                                                                    isSelected
                                                                        ? '2px solid rgba(0,0,0,0.5)'
                                                                        : undefined,
                                                                fontWeight:
                                                                    isSelected
                                                                        ? 'bold'
                                                                        : 'normal',
                                                            }}
                                                            onClick={() =>
                                                                handleRiskFilter(
                                                                    level
                                                                )
                                                            }
                                                        >
                                                            {level}: {count}

                                                            {isSelected && (
                                                                <CheckCircleOutlined
                                                                    style={{
                                                                        marginLeft: 4,
                                                                    }}
                                                                />
                                                            )}
                                                        </Tag>
                                                    );
                                                }
                                            )}

                                            {selectedRiskFilter && (
                                                <Button
                                                    size="small"
                                                    type="link"
                                                    onClick={() =>
                                                        setSelectedRiskFilter(
                                                            null
                                                        )
                                                    }
                                                >
                                                    Clear
                                                </Button>
                                            )}
                                        </div>

                                        <Progress
                                            percent={
                                                batchMeta.total_transactions
                                                    ? ((batchMeta.fraud_count ||
                                                          0) /
                                                          batchMeta.total_transactions) *
                                                      100
                                                    : 0
                                            }
                                            status="exception"
                                            showInfo={false}
                                            strokeWidth={8}
                                        />
                                    </div>
                                </>
                            )}

                            <Divider
                                style={{
                                    margin: '12px 0',
                                }}
                            />

                            <Table
                                dataSource={
                                    filteredBatchResults || undefined
                                }
                                columns={batchColumns}
                                rowKey="index"
                                size="small"
                                pagination={{
                                    pageSize: 10,
                                }}
                                scroll={{
                                    y: 300,
                                }}
                            />
                        </Card>
                    ) : (
                        modelLoaded && (
                            <Card
                                style={{
                                    textAlign: 'center',
                                    padding: 32,
                                }}
                            >
                                <UploadOutlined
                                    style={{
                                        fontSize: 36,
                                        color: '#d9d9d9',
                                    }}
                                />

                                <Title
                                    level={5}
                                    type="secondary"
                                    style={{
                                        marginTop: 12,
                                    }}
                                >
                                    Upload data and run batch inference
                                </Title>
                            </Card>
                        )
                    )}
                </Col>
            </Row>
        </div>
    );
}