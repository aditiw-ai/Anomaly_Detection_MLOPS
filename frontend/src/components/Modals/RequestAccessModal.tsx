import React, { useState } from 'react';
import { Modal, Form, Select, Input, Button, message, Alert, Typography } from 'antd';
import { SecurityScanOutlined } from '@ant-design/icons';
import { api } from '../../api/axios';
import { useAuth } from '../../contexts/AuthContext';

const { Text } = Typography;
const { Option } = Select;

const ROLES = [
  { value: 'ML_ENGINEER', label: 'ML Engineer', desc: 'Train models, manage jobs, and delete experiments.' },
  { value: 'DATA_ENGINEER', label: 'Data Engineer', desc: 'Upload, manage, and delete datasets.' },
  { value: 'DEPLOYER', label: 'Deployer', desc: 'Deploy models and configure monitoring.' },
  { value: 'ADMIN', label: 'Admin', desc: 'Full system access and user management.' },
];

interface RequestAccessModalProps {
  open: boolean;
  onClose: () => void;
}

export const RequestAccessModal: React.FC<RequestAccessModalProps> = ({ open, onClose }) => {
  const { token } = useAuth();
  const [loading, setLoading] = useState(false);
  const [form] = Form.useForm();

  const handleFinish = async (values: any) => {
    setLoading(true);
    try {
      await api.post('/role-requests/', {
        requested_role: values.requested_role,
        reason: values.reason,
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      message.success('Request submitted successfully! An admin will review it soon.');
      form.resetFields();
      onClose();
    } catch (error: any) {
      message.error(error.response?.data?.detail || 'Failed to submit request');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal
      title={
        <span>
          <SecurityScanOutlined style={{ marginRight: 8, color: '#1890ff' }} />
          Request Higher Access
        </span>
      }
      open={open}
      onCancel={() => { form.resetFields(); onClose(); }}
      footer={null}
      width={480}
    >
      <Alert
        message="Elevate Your Permissions"
        description="Select the role you need and provide a brief justification. An administrator will review your request."
        type="info"
        showIcon
        style={{ marginBottom: 20 }}
      />

      <Form form={form} layout="vertical" onFinish={handleFinish}>
        <Form.Item
          name="requested_role"
          label="Target Role"
          rules={[{ required: true, message: 'Please select a role' }]}
        >
          <Select placeholder="Select a role" size="large">
            {ROLES.map(role => (
              <Option key={role.value} value={role.value}>
                <div>
                  <Text strong>{role.label}</Text>
                  <br />
                  <Text type="secondary" style={{ fontSize: 12 }}>{role.desc}</Text>
                </div>
              </Option>
            ))}
          </Select>
        </Form.Item>

        <Form.Item
          name="reason"
          label="Justification"
          rules={[{ required: true, message: 'Please provide a reason' }]}
        >
          <Input.TextArea
            placeholder="Why do you need this role? (e.g., 'Need to start a training job for the anomaly detection project')"
            rows={4}
          />
        </Form.Item>

        <Form.Item style={{ marginBottom: 0, marginTop: 12 }}>
          <Button type="primary" htmlType="submit" loading={loading} block size="large">
            Submit Request
          </Button>
        </Form.Item>
      </Form>
    </Modal>
  );
};
