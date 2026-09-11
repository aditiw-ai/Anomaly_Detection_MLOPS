/**
 * Placeholder Pages
 * These will be implemented in subsequent sprints.
 */
import { Card, Typography, Empty, Button } from 'antd';
import { ExperimentOutlined, AppstoreOutlined, LineChartOutlined, AlertOutlined } from '@ant-design/icons';

const { Title, Text } = Typography;

// Training Page (Sprint 2)
export function Training() {
    return (
        <div className="fade-in">
            <div className="page-header">
                <div>
                    <Title level={2} style={{ margin: 0 }}>Training</Title>
                    <Text type="secondary">Configure and run model training jobs</Text>
                </div>
                <Button type="primary" icon={<ExperimentOutlined />}>New Training Job</Button>
            </div>
            <Card>
                <Empty
                    image={<ExperimentOutlined style={{ fontSize: 64, color: '#d9d9d9' }} />}
                    description={
                        <span>
                            <Text strong style={{ display: 'block', marginBottom: 8 }}>Training Module</Text>
                            <Text type="secondary">Coming in Sprint 2 - Feature engineering and model training</Text>
                        </span>
                    }
                />
            </Card>
        </div>
    );
}

// Model Registry Page (Sprint 3)
export function ModelRegistry() {
    return (
        <div className="fade-in">
            <div className="page-header">
                <div>
                    <Title level={2} style={{ margin: 0 }}>Model Registry</Title>
                    <Text type="secondary">View, compare, and promote models</Text>
                </div>
            </div>
            <Card>
                <Empty
                    image={<AppstoreOutlined style={{ fontSize: 64, color: '#d9d9d9' }} />}
                    description={
                        <span>
                            <Text strong style={{ display: 'block', marginBottom: 8 }}>Model Registry</Text>
                            <Text type="secondary">Coming in Sprint 3 - Model versioning and promotion</Text>
                        </span>
                    }
                />
            </Card>
        </div>
    );
}

// Monitoring Page (Sprint 4)
export function Monitoring() {
    return (
        <div className="fade-in">
            <div className="page-header">
                <div>
                    <Title level={2} style={{ margin: 0 }}>Monitoring</Title>
                    <Text type="secondary">Track drift, performance, and bias metrics</Text>
                </div>
            </div>
            <Card>
                <Empty
                    image={<LineChartOutlined style={{ fontSize: 64, color: '#d9d9d9' }} />}
                    description={
                        <span>
                            <Text strong style={{ display: 'block', marginBottom: 8 }}>Monitoring Dashboard</Text>
                            <Text type="secondary">Coming in Sprint 4 - Drift detection and bias monitoring</Text>
                        </span>
                    }
                />
            </Card>
        </div>
    );
}

// Alerts Page (Sprint 5)
export function Alerts() {
    return (
        <div className="fade-in">
            <div className="page-header">
                <div>
                    <Title level={2} style={{ margin: 0 }}>Alerts</Title>
                    <Text type="secondary">View and manage active alerts</Text>
                </div>
            </div>
            <Card>
                <Empty
                    image={<AlertOutlined style={{ fontSize: 64, color: '#d9d9d9' }} />}
                    description={
                        <span>
                            <Text strong style={{ display: 'block', marginBottom: 8 }}>Alerts Center</Text>
                            <Text type="secondary">Coming in Sprint 5 - Alerting and notifications</Text>
                        </span>
                    }
                />
            </Card>
        </div>
    );
}
