import React from 'react';
import { Button, Row, Col, Typography, Space, Divider } from 'antd';
import {
    SafetyCertificateOutlined,
    NodeIndexOutlined,
    ThunderboltOutlined,
    DatabaseOutlined,
    AlertOutlined,
    EyeOutlined,
    SyncOutlined,
    LockOutlined,
    ArrowRightOutlined
} from '@ant-design/icons';
import './landing.css';

const { Title, Text, Paragraph } = Typography;
const HERO_IMAGE_PRIMARY = '/illustrations/hero-risk.jpg';
const HERO_IMAGE_FALLBACK = '/illustrations/workflow-map.svg';
const featureCards = [
    {
        title: 'Anomaly-First Intelligence',
        description: 'Purpose-built for anomaly detection and MLOps, not a generic MLOps dashboard retrofitted for banking.',
        icon: <SafetyCertificateOutlined className="panel-icon" />,
    },
    {
        title: 'Signal + Graph Correlation',
        description: 'Correlate transactions, entities, and model outputs in one decision view, a gap in most platforms.',
        icon: <NodeIndexOutlined className="panel-icon" />,
    },
    {
        title: 'Instant Circuit Breakers',
        description: 'Trigger policy actions in real time when anomaly risk spikes, not just passive monitoring alerts.',
        icon: <ThunderboltOutlined className="panel-icon" />,
    },
    {
        title: 'Drift to Decision',
        description: 'Translate drift directly into operational playbooks so teams can act before losses compound.',
        icon: <AlertOutlined className="panel-icon" />,
    },
    {
        title: 'Explainable Risk Trails',
        description: 'Track why a score changed across features and thresholds for audits, regulators, and trust teams.',
        icon: <EyeOutlined className="panel-icon" />,
    },
    {
        title: 'Adaptive Retraining Loops',
        description: 'Close the loop between production outcomes and retraining so models continuously learn from live anomaly patterns.',
        icon: <SyncOutlined className="panel-icon" />,
    },
    {
        title: 'Secure Multi-Role Controls',
        description: 'Role-aware workflows for risk, ML, and compliance teams working together without permission sprawl.',
        icon: <LockOutlined className="panel-icon" />,
    },
    {
        title: 'Stack Ready',
        description: 'Deploy with FastAPI, React, Docker, and modern data pipelines while preserving anomaly-specific controls.',
        icon: <DatabaseOutlined className="panel-icon" />,
        highlight: true,
    },
];

const LandingPage: React.FC = () => {
    const handleGetStarted = () => {
        window.dispatchEvent(new Event('shadow-hubble:open-auth'));
    };

    return (
        <div className="landing-container">
            {/* 1. Hero Section */}
            <section className="hero-section">
                <Row gutter={[48, 48]} align="middle" className="full-height-row">
                    <Col xs={24} lg={11}>
                        <Space direction="vertical" size={24}>
                            <Title level={1} className="hero-title">
                                Detect ML Risk <br />
                                <span className="text-accent">Before it Escalates.</span>
                            </Title>
                            <Paragraph className="hero-subtitle">
                                The specialized MLOps platform for anomaly detection teams.
                                Monitor model behavior, identify drift, and orchestrate
                                responses in real-time without the dashboard clutter.
                            </Paragraph>
                            <Space size={16}>
                                <Button
                                    type="primary"
                                    size="large"
                                    icon={<ArrowRightOutlined />}
                                    className="cta-button"
                                    onClick={handleGetStarted}
                                >
                                    Get Started
                                </Button>
                                <Button size="large" className="secondary-button">
                                    View Docs
                                </Button>
                            </Space>
                        </Space>
                    </Col>
                    <Col xs={24} lg={13} className="hero-visual-container">
                        <img
                            src={HERO_IMAGE_PRIMARY}
                            alt="ML Risk Flow Illustration"
                            className="hero-illustration"
                            onError={(e) => {
                                const target = e.currentTarget;
                                if (target.src.includes(HERO_IMAGE_FALLBACK)) return;
                                target.src = HERO_IMAGE_FALLBACK;
                            }}
                        />
                    </Col>
                </Row>
            </section>

            {/* 2. The Eight Panels (Main Body) */}
            <section className="panels-section">
                <Row gutter={[24, 24]}>
                    {featureCards.map((card) => (
                        <Col xs={24} md={12} lg={6} key={card.title}>
                            <div className={`info-panel${card.highlight ? ' highlight-panel' : ''}`}>
                                {card.icon}
                                <Title level={4}>{card.title}</Title>
                                <Text type="secondary">{card.description}</Text>
                                {card.highlight ? (
                                    <>
                                        <Divider className="panel-divider" />
                                        <Button type="link" className="panel-link">
                                            Explore Use Cases <ArrowRightOutlined />
                                        </Button>
                                    </>
                                ) : null}
                            </div>
                        </Col>
                    ))}
                </Row>
            </section>

            {/* 4. Footer Trust Band */}
            <footer className="footer-band">
                <Row justify="space-between" align="middle">
                    <Col>
                        <Text className="footer-copy">© 2026 AnomaLens. Enterprise-Grade Risk Intelligence.</Text>
                    </Col>
                    <Col>
                        <Space split={<Divider type="vertical" />}>
                            <Text className="footer-link">Privacy</Text>
                            <Text className="footer-link">Terms</Text>
                            <Text className="footer-link">Security</Text>
                        </Space>
                    </Col>
                </Row>
            </footer>
        </div>
    );
};

export default LandingPage;
