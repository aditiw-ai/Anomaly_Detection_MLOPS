import React, { useEffect, useState } from 'react';
import {
  Table, Select, Button, Tag, Popconfirm, message,
  Card, Typography, Space, Badge, Modal, Form, Input, Tooltip, Tabs, Descriptions
} from 'antd';
import {
  DeleteOutlined, CrownOutlined, PlusOutlined, CheckOutlined,
  CloseOutlined, HourglassOutlined, TeamOutlined, UserOutlined,
  SettingOutlined
} from '@ant-design/icons';
import { api } from '../api/axios';
import { useAuth } from '../contexts/AuthContext';
import { RequestAccessModal } from '../components/Modals/RequestAccessModal';

const { Title, Text } = Typography;
const { Option } = Select;

const ROLES = ['ADMIN', 'DATA_ENGINEER', 'ML_ENGINEER', 'DEPLOYER', 'VIEWER'];

const ROLE_COLORS: Record<string, string> = {
  ADMIN: 'red',
  DATA_ENGINEER: 'blue',
  ML_ENGINEER: 'purple',
  DEPLOYER: 'green',
  VIEWER: 'default',
};

interface UserRecord {
  id: string;
  name: string;
  email: string;
  roles: string[];
  is_active: boolean;
}

interface RoleRequest {
  id: string;
  user_id: string;
  user_name?: string;
  user_email?: string;
  requested_role: string;
  reason: string;
  status: 'PENDING' | 'APPROVED' | 'REJECTED';
  admin_notes?: string;
  created_at: string;
}

interface UserProfile {
  id: string;
  name: string;
  email: string;
  roles: string[];
  is_active: boolean;
  created_at: string;
}

export const Settings: React.FC = () => {
  const { token, user: currentUser, hasRole, refreshUser } = useAuth();
  const isAdmin = hasRole(['ADMIN']);
  const [users, setUsers] = useState<UserRecord[]>([]);
  const [requests, setRequests] = useState<RoleRequest[]>([]);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const [requestsLoading, setRequestsLoading] = useState(false);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [requestAccessModalOpen, setRequestAccessModalOpen] = useState(false);
  const [createLoading, setCreateLoading] = useState(false);
  const [form] = Form.useForm();

  const headers = { Authorization: `Bearer ${token}` };

  const fetchUsers = async () => {
    setLoading(true);
    try {
      const res = await api.get('/admin/users', { headers });
      setUsers(res.data);
    } catch {
      message.error('Failed to load users');
    } finally {
      setLoading(false);
    }
  };

  const fetchRequests = async () => {
    setRequestsLoading(true);
    try {
      const endpoint = isAdmin ? '/role-requests/admin/list' : '/role-requests/me';
      const res = await api.get(endpoint, { headers });
      setRequests(res.data);
    } catch {
      message.error('Failed to load requests');
    } finally {
      setRequestsLoading(false);
    }
  };

  const fetchUserProfile = async () => {
    try {
      const res = await api.get('/auth/me', { headers });
      setProfile(res.data);
    } catch {
      message.error('Failed to load profile');
    }
  };

  useEffect(() => {
    if (token) {
      if (isAdmin) {
        fetchUsers();
        fetchRequests();
      } else {
        fetchRequests();
        fetchUserProfile();
      }
    }
  }, [isAdmin, token]);

  const handleCreateUser = async (values: any) => {
    setCreateLoading(true);
    try {
      await api.post('/admin/users', {
        name: values.name,
        email: values.email,
        password: values.password,
        roles: [values.role],
      }, { headers });
      message.success(`User "${values.name}" created as ${values.role}`);
      form.resetFields();
      setCreateModalOpen(false);
      fetchUsers();
    } catch (e: any) {
      message.error(e.response?.data?.detail || 'Failed to create user');
    } finally {
      setCreateLoading(false);
    }
  };

  const handleApproveRequest = async (requestId: string) => {
    try {
      await api.patch(`/role-requests/admin/${requestId}/approve`, {}, { headers });
      message.success('Request approved and user role updated');
      fetchRequests();
      if (isAdmin) fetchUsers();
      refreshUser();
    } catch (e: any) {
      message.error(e.response?.data?.detail || 'Failed to approve request');
    }
  };

  const handleRejectRequest = async (requestId: string) => {
    try {
      await api.patch(`/role-requests/admin/${requestId}/reject`, {}, { headers });
      message.success('Request rejected');
      fetchRequests();
    } catch (e: any) {
      message.error(e.response?.data?.detail || 'Failed to reject request');
    }
  };

  const handleRoleChange = async (userId: string, newRole: string) => {
    try {
      await api.patch(`/admin/users/${userId}/role`, { roles: [newRole] }, { headers });
      message.success('Role updated');
      fetchUsers();
      refreshUser();
    } catch (e: any) {
      message.error(e.response?.data?.detail || 'Failed to update role');
    }
  };

  const handleToggleStatus = async (userId: string) => {
    try {
      await api.patch(`/admin/users/${userId}/status`, {}, { headers });
      message.success('Status updated');
      fetchUsers();
    } catch {
      message.error('Failed to update status');
    }
  };

  const handleDelete = async (userId: string) => {
    try {
      await api.delete(`/admin/users/${userId}`, { headers });
      message.success('User deleted');
      fetchUsers();
    } catch {
      message.error('Failed to delete user');
    }
  };

  const userColumns = [
    {
      title: 'User',
      key: 'user',
      render: (_: any, record: UserRecord) => (
        <Space>
          <div style={{
            width: 36, height: 36, borderRadius: '50%',
            backgroundColor: record.roles[0] === 'ADMIN' ? '#ef4444'
              : record.roles[0] === 'DATA_ENGINEER' ? '#3b82f6'
              : record.roles[0] === 'ML_ENGINEER' ? '#8b5cf6'
              : record.roles[0] === 'DEPLOYER' ? '#10b981'
              : '#6b7280',
            color: 'white', display: 'flex', alignItems: 'center',
            justifyContent: 'center', fontWeight: 700, flexShrink: 0,
          }}>
            {record.name?.charAt(0)?.toUpperCase() || 'U'}
          </div>
          <div>
            <div style={{ fontWeight: 600 }}>
              {record.name}
              {record.id === currentUser?.id && (
                <Tag color="blue" style={{ marginLeft: 8, fontSize: 10 }}>You</Tag>
              )}
            </div>
            <div style={{ color: '#888', fontSize: 12 }}>{record.email}</div>
          </div>
        </Space>
      ),
    },
    {
      title: 'Role',
      key: 'role',
      render: (_: any, record: UserRecord) => (
        <Tooltip title={!isAdmin ? 'Only admins can change roles' : ''}>
          <Select
            value={record.roles[0] || 'VIEWER'}
            style={{ width: 160 }}
            disabled={record.id === currentUser?.id || !isAdmin}
            onChange={(val) => handleRoleChange(record.id, val)}
          >
            {ROLES.map(r => (
              <Option key={r} value={r}>
                <Tag color={ROLE_COLORS[r]}>{r}</Tag>
              </Option>
            ))}
          </Select>
        </Tooltip>
      ),
    },
    {
      title: 'Status',
      key: 'status',
      render: (_: any, record: UserRecord) => (
        <Badge
          status={record.is_active ? 'success' : 'error'}
          text={record.is_active ? 'Active' : 'Inactive'}
        />
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: UserRecord) => (
        <Space>
          <Tooltip title={!isAdmin ? 'Only admins can manage users' : ''}>
            <Button
              size="small"
              onClick={() => handleToggleStatus(record.id)}
              disabled={record.id === currentUser?.id || !isAdmin}
            >
              {record.is_active ? 'Deactivate' : 'Activate'}
            </Button>
          </Tooltip>
          <Popconfirm
            title="Delete this user?"
            description="This action cannot be undone."
            onConfirm={() => handleDelete(record.id)}
            disabled={record.id === currentUser?.id || !isAdmin}
          >
            <Tooltip title={!isAdmin ? 'Only admins can delete users' : ''}>
              <Button
                size="small"
                danger
                icon={<DeleteOutlined />}
                disabled={record.id === currentUser?.id || !isAdmin}
              />
            </Tooltip>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  const requestColumns = [
    {
      title: 'Requested By',
      key: 'user',
      hidden: !isAdmin,
      render: (_: any, record: RoleRequest) => (
        <div>
          <div style={{ fontWeight: 600 }}>{record.user_name}</div>
          <div style={{ fontSize: 12, color: '#888' }}>{record.user_email}</div>
        </div>
      ),
    },
    {
      title: 'Target Role',
      key: 'role',
      render: (_role: string, record: RoleRequest) => (
        <Tag color={ROLE_COLORS[record.requested_role]}>{record.requested_role}</Tag>
      ),
    },
    {
      title: 'Justification',
      dataIndex: 'reason',
      key: 'reason',
      width: 300,
      render: (text: string) => <Text type="secondary" style={{ fontSize: 13 }}>{text}</Text>,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Badge
          status={status === 'PENDING' ? 'processing' : status === 'APPROVED' ? 'success' : 'error'}
          text={status}
        />
      ),
    },
    {
      title: 'Admin Notes',
      dataIndex: 'admin_notes',
      key: 'admin_notes',
      render: (text: string) =>
        text ? <Text type="secondary" style={{ fontSize: 13 }}>{text}</Text> : <Text type="secondary" italic>-</Text>,
    },
    {
      title: 'Actions',
      key: 'actions',
      hidden: !isAdmin,
      render: (_: any, record: RoleRequest) =>
        record.status === 'PENDING' && isAdmin ? (
          <Space>
            <Button
              type="primary"
              size="small"
              icon={<CheckOutlined />}
              onClick={() => handleApproveRequest(record.id)}
            >
              Approve
            </Button>
            <Button
              danger
              size="small"
              icon={<CloseOutlined />}
              onClick={() => handleRejectRequest(record.id)}
            >
              Reject
            </Button>
          </Space>
        ) : null,
    },
  ].filter(column => !column.hidden);

  // --- Profile section for non-admins ---
  const profileContent = profile ? (
    <div style={{ padding: '20px 0' }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: 32, gap: 24 }}>
        <div style={{
          width: 80, height: 80, borderRadius: '50%',
          backgroundColor:
            ROLE_COLORS[profile.roles[0]] === 'red' ? '#ef4444'
            : ROLE_COLORS[profile.roles[0]] === 'blue' ? '#3b82f6'
            : ROLE_COLORS[profile.roles[0]] === 'purple' ? '#8b5cf6'
            : ROLE_COLORS[profile.roles[0]] === 'green' ? '#10b981'
            : '#6b7280',
          color: 'white', display: 'flex', alignItems: 'center',
          justifyContent: 'center', fontSize: 32, fontWeight: 700,
          boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
        }}>
          {profile.name?.charAt(0)?.toUpperCase()}
        </div>
        <div>
          <Title level={4} style={{ margin: 0 }}>{profile.name}</Title>
          <Text type="secondary">{profile.email}</Text>
          <div style={{ marginTop: 8 }}>
            {profile.roles.map(r => <Tag color={ROLE_COLORS[r]} key={r}>{r}</Tag>)}
          </div>
        </div>
      </div>

      <Descriptions
        bordered
        column={1}
        size="middle"
        labelStyle={{ width: 180, fontWeight: 600, backgroundColor: '#fafafa' }}
      >
        <Descriptions.Item label="Account ID">
          <Text code copyable>{profile.id}</Text>
        </Descriptions.Item>
        <Descriptions.Item label="Status">
          <Badge status={profile.is_active ? 'success' : 'error'} text={profile.is_active ? 'Active' : 'Inactive'} />
        </Descriptions.Item>
        <Descriptions.Item label="Email Address">{profile.email}</Descriptions.Item>
        <Descriptions.Item label="Member Since">
          {new Date(profile.created_at).toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' })}
        </Descriptions.Item>
      </Descriptions>
    </div>
  ) : (
    <div style={{ textAlign: 'center', padding: 40 }}>
      <HourglassOutlined spin style={{ fontSize: 24 }} />
    </div>
  );

  // --- Tabs for the User Management section ---
  const userManagementTabs = isAdmin
    ? [
        {
          key: 'users',
          label: (
            <span>
              <TeamOutlined /> All Users
            </span>
          ),
          children: (
            <Table
              dataSource={users}
              columns={userColumns}
              rowKey="id"
              loading={loading}
              pagination={{ pageSize: 12 }}
            />
          ),
        },
        {
          key: 'requests',
          label: (
            <span>
              <Badge
                count={requests.filter((r: RoleRequest) => r.status === 'PENDING').length}
                offset={[10, 0]}
                size="small"
              >
                <HourglassOutlined />
              </Badge>
              <span style={{ marginLeft: 8 }}>Role Requests</span>
            </span>
          ),
          children: (
            <Table
              dataSource={requests}
              columns={requestColumns}
              rowKey="id"
              loading={requestsLoading}
              pagination={{ pageSize: 12 }}
            />
          ),
        },
      ]
    : [
        {
          key: 'profile',
          label: (
            <span>
              <UserOutlined /> My Profile
            </span>
          ),
          children: <div style={{ maxWidth: 800 }}>{profileContent}</div>,
        },
        {
          key: 'my-requests',
          label: (
            <span>
              <HourglassOutlined /> My Role Requests
            </span>
          ),
          children: (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={() => setRequestAccessModalOpen(true)}
                >
                  Request Higher Access
                </Button>
              </div>
              <Table
                dataSource={requests}
                columns={requestColumns}
                rowKey="id"
                loading={requestsLoading}
                pagination={{ pageSize: 12 }}
              />
            </div>
          ),
        },
      ];

  // --- Top-level Settings page tabs ---
  const settingsTabs = [
    {
      key: 'user-management',
      label: (
        <span>
          {isAdmin ? <CrownOutlined /> : <UserOutlined />}
          {' '}
          {isAdmin ? 'User Management' : 'Account & Permissions'}
        </span>
      ),
      children: (
        <div>
          {/* Sub-header with action button */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <Text type="secondary">
              {isAdmin
                ? 'Manage platform users, roles, and access requests.'
                : 'View your profile and manage permission requests.'}
            </Text>
            <Tooltip title={!isAdmin ? 'Only admins can create users' : ''}>
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => setCreateModalOpen(true)}
                disabled={!isAdmin}
              >
                Create User
              </Button>
            </Tooltip>
          </div>
          <Tabs items={userManagementTabs} />
        </div>
      ),
    },
  ];

  return (
    <div>
      {/* Page Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 24 }}>
        <SettingOutlined style={{ fontSize: 24, color: '#2563EB' }} />
        <Title level={3} style={{ margin: 0 }}>Settings</Title>
      </div>

      <Card>
        <Tabs items={settingsTabs} tabPosition="left" style={{ minHeight: 400 }} />
      </Card>

      {/* Create User Modal */}
      <Modal
        title="Create New User"
        open={createModalOpen}
        onCancel={() => { setCreateModalOpen(false); form.resetFields(); }}
        footer={null}
        width={440}
      >
        <Form form={form} layout="vertical" onFinish={handleCreateUser} style={{ marginTop: 16 }}>
          <Form.Item name="name" label="Full Name" rules={[{ required: true }]}>
            <Input placeholder="e.g. Jane Smith" size="large" />
          </Form.Item>
          <Form.Item name="email" label="Email" rules={[{ required: true }, { type: 'email' }]}>
            <Input placeholder="jane@company.com" size="large" />
          </Form.Item>
          <Form.Item name="password" label="Password" rules={[{ required: true, min: 6, message: 'Minimum 6 characters' }]}>
            <Input.Password placeholder="Temporary password" size="large" />
          </Form.Item>
          <Form.Item name="role" label="Role" initialValue="VIEWER" rules={[{ required: true }]}>
            <Select size="large">
              {ROLES.map(r => (
                <Option key={r} value={r}>
                  <Tag color={ROLE_COLORS[r]}>{r}</Tag>
                </Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item style={{ marginBottom: 0, marginTop: 8 }}>
            <Button type="primary" htmlType="submit" loading={createLoading} block size="large">
              Create User
            </Button>
          </Form.Item>
        </Form>
      </Modal>

      <RequestAccessModal
        open={requestAccessModalOpen}
        onClose={() => {
          setRequestAccessModalOpen(false);
          fetchRequests();
        }}
      />
    </div>
  );
};
