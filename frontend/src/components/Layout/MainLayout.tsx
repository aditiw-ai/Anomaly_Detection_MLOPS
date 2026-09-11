/**
 * Main Layout Component
 * Sidebar navigation with header and content area.
 * Sidebar items are filtered based on the current user's role.
 */
import { useEffect, useState } from 'react';
import { Layout, Menu, Button, Dropdown, Typography } from 'antd';
const { Text } = Typography;
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { LoginModal } from '../Auth/LoginModal';
import {
    DashboardOutlined,
    DatabaseOutlined,
    ExperimentOutlined,
    AppstoreOutlined,
    LineChartOutlined,
    AlertOutlined,
    MenuFoldOutlined,
    MenuUnfoldOutlined,
    ThunderboltOutlined,
    UserOutlined,
} from '@ant-design/icons';
import type { ReactNode } from 'react';

const { Sider, Content, Header } = Layout;

interface MainLayoutProps {
    children: ReactNode;
}

// Map each nav item to the roles that can see it
const allMenuItems = [
    { key: '/', icon: <DashboardOutlined />, label: 'Dashboard', roles: null }, // visible to all
    { key: '/data', icon: <DatabaseOutlined />, label: 'Datasets', roles: ['ADMIN', 'DATA_ENGINEER', 'ML_ENGINEER', 'VIEWER'] },
    { key: '/training', icon: <ExperimentOutlined />, label: 'Training Jobs', roles: ['ADMIN', 'ML_ENGINEER', 'VIEWER'] },
    { key: '/models', icon: <AppstoreOutlined />, label: 'Models', roles: ['ADMIN', 'DATA_ENGINEER', 'ML_ENGINEER', 'DEPLOYER', 'VIEWER'] },
    { key: '/inference', icon: <ThunderboltOutlined />, label: 'Inference', roles: ['ADMIN', 'DEPLOYER', 'VIEWER'] },
    { key: '/monitoring', icon: <LineChartOutlined />, label: 'Monitoring', roles: null }, // visible to all logged in
    { key: '/alerts', icon: <AlertOutlined />, label: 'Alerts', roles: null },
];

export function MainLayout({ children }: MainLayoutProps) {
    const [collapsed, setCollapsed] = useState(false);
    const [loginModalOpen, setLoginModalOpen] = useState(false);
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const location = useLocation();

    useEffect(() => {
        const handleOpenAuth = () => setLoginModalOpen(true);
        window.addEventListener('shadow-hubble:open-auth', handleOpenAuth);
        return () => window.removeEventListener('shadow-hubble:open-auth', handleOpenAuth);
    }, []);

    const userRoles: string[] = user?.roles || [];
    
    // All menu items are visible to all logged-in users
    const menuItems = allMenuItems.filter(() => {
        if (!user) return false;      // hidden when not logged in
        return true;                  // visible to all users regardless of role
    });

    const roleColors: Record<string, string> = {
        ADMIN: '#ef4444',
        DATA_ENGINEER: '#3b82f6',
        ML_ENGINEER: '#8b5cf6',
        DEPLOYER: '#10b981',
        VIEWER: '#6b7280',
    };
    const primaryRole = userRoles[0] || '';
    const avatarBg = roleColors[primaryRole] || '#2563EB';
    const isAuthenticated = !!user;

    const handleLogout = () => {
        logout();
        navigate('/');
    };

    if (!isAuthenticated) {
        return (
            <Layout style={{ minHeight: '100vh', background: '#F8FAFC' }}>
                <Header
                    style={{
                        background: '#fff',
                        padding: '0 24px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        boxShadow: '0 1px 4px rgba(0, 0, 0, 0.05)',
                        position: 'sticky',
                        top: 0,
                        zIndex: 99,
                    }}
                >
                    <span style={{ fontSize: 18, fontWeight: 700, color: '#2563EB' }}>AnomaLens</span>
                    <Button type="primary" onClick={() => setLoginModalOpen(true)}>
                        Sign In
                    </Button>
                </Header>

                <Content
                    style={{
                        padding: '0 24px',
                        background: '#F8FAFC',
                        minHeight: 'calc(100vh - 64px)',
                    }}
                >
                    {children}
                </Content>

                <LoginModal
                    open={loginModalOpen}
                    onCancel={() => setLoginModalOpen(false)}
                    onSuccess={() => setLoginModalOpen(false)}
                />
            </Layout>
        );
    }

    return (
        <Layout style={{ minHeight: '100vh' }}>
            <Sider
                collapsible
                collapsed={collapsed}
                onCollapse={setCollapsed}
                trigger={null}
                theme="light"
                style={{
                    boxShadow: '2px 0 8px rgba(0, 0, 0, 0.05)',
                    position: 'fixed',
                    height: '100vh',
                    left: 0,
                    top: 0,
                    bottom: 0,
                    zIndex: 100,
                }}
            >
                <div
                    style={{
                        height: 64,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: collapsed ? 'center' : 'flex-start',
                        padding: collapsed ? 0 : '0 16px',
                        borderBottom: '1px solid #f0f0f0',
                    }}
                >
                    {collapsed ? (
                        <span style={{ fontSize: 24 }}>🔮</span>
                    ) : (
                        <span style={{ fontSize: 18, fontWeight: 700, color: '#2563EB' }}>
                            AnomaLens
                        </span>
                    )}
                </div>
                <Menu
                    mode="inline"
                    selectedKeys={[location.pathname]}
                    items={menuItems}
                    onClick={({ key }) => navigate(key)}
                    style={{ borderRight: 0 }}
                />
            </Sider>

            <Layout style={{ marginLeft: collapsed ? 80 : 200, transition: 'margin-left 0.2s' }}>
                <Header
                    style={{
                        background: '#fff',
                        padding: '0 24px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        boxShadow: '0 1px 4px rgba(0, 0, 0, 0.05)',
                        position: 'sticky',
                        top: 0,
                        zIndex: 99,
                    }}
                >
                    <div onClick={() => setCollapsed(!collapsed)} style={{ cursor: 'pointer', fontSize: 18 }}>
                        {collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                        <AlertOutlined style={{ fontSize: 18 }} />
                        {user ? (
                            <Dropdown menu={{
                                items: [
                                    {
                                        key: 'email',
                                        label: <Text type="secondary" style={{ fontSize: 12 }}>{user.email}</Text>,
                                        disabled: true,
                                    },
                                    { type: 'divider' },
                                    {
                                        key: 'profile',
                                        label: 'My Account',
                                        icon: <UserOutlined />,
                                        onClick: () => navigate('/settings')
                                    },
                                    { type: 'divider' },
                                    { key: 'logout', label: 'Sign Out', onClick: handleLogout, danger: true }
                                ]
                            }} placement="bottomRight">
                                <div style={{
                                    width: 34, height: 34,
                                    borderRadius: '50%',
                                    background: avatarBg,
                                    color: 'white',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    fontWeight: 700,
                                    cursor: 'pointer',
                                    border: `2px solid ${avatarBg}`,
                                    boxShadow: `0 0 0 2px ${avatarBg}33`,
                                }}>
                                    {user.name ? user.name.charAt(0).toUpperCase() : 'U'}
                                </div>
                            </Dropdown>
                        ) : (
                            <Button type="primary" onClick={() => setLoginModalOpen(true)}>
                                Sign In
                            </Button>
                        )}
                    </div>
                </Header>

                <Content
                    style={{
                        padding: 24,
                        background: '#f5f5f5',
                        minHeight: 'calc(100vh - 64px)',
                    }}
                >
                    {children}
                </Content>
            </Layout>

            <LoginModal
                open={loginModalOpen}
                onCancel={() => setLoginModalOpen(false)}
                onSuccess={() => setLoginModalOpen(false)}
            />
        </Layout>
    );
}
