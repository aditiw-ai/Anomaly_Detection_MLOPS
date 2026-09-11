import React, { useState } from 'react';
import { Modal, Form, Input, Button, message, Tabs } from 'antd';
import { UserOutlined, LockOutlined, MailOutlined, IdcardOutlined } from '@ant-design/icons';
import { api } from '../../api/axios';
import { useAuth } from '../../contexts/AuthContext';

interface LoginModalProps {
  open: boolean;
  onCancel: () => void;
  onSuccess: () => void;
}

export const LoginModal: React.FC<LoginModalProps> = ({ open, onCancel, onSuccess }) => {
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('login');
  const { login } = useAuth();
  const [loginForm] = Form.useForm();
  const [signupForm] = Form.useForm();

  const handleLogin = async (values: any) => {
    setLoading(true);
    try {
      const email = String(values.email || '').trim().toLowerCase();
      const password = String(values.password || '');
      const formData = new URLSearchParams();
      formData.append('username', email);
      formData.append('password', password);

      const response = await api.post('/auth/login', formData, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
      });

      const { access_token, user_id, name, roles } = response.data;
      login(access_token, { id: user_id, email, name: name || 'User', roles });

      message.success('Successfully logged in!');
      loginForm.resetFields();
      onSuccess();
    } catch (error: any) {
      console.error('Login failed', error);
      if (!error.response) {
        message.error('Cannot reach backend API. Please ensure backend is running on port 8000.');
      } else {
        const detail =
          error.response?.data?.detail ||
          (typeof error.response?.data === 'string' ? error.response.data : '') ||
          `Login failed (HTTP ${error.response?.status || 'unknown'})`;
        message.error(detail);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleSignup = async (values: any) => {
    if (values.password !== values.confirmPassword) {
      message.error("Passwords don't match!");
      return;
    }

    setLoading(true);
    try {
      const response = await api.post('/auth/signup', {
        email: values.email,
        password: values.password,
        name: values.name
      });

      const { access_token, user_id, name, roles } = response.data;
      login(access_token, { id: user_id, email: values.email, name, roles });

      message.success('Account created successfully!');
      signupForm.resetFields();
      onSuccess();
    } catch (error: any) {
      console.error('Signup failed', error);
      message.error(error.response?.data?.detail || 'Signup failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const loginTab = (
    <div style={{ paddingTop: '10px' }}>
      <Form form={loginForm} layout="vertical" onFinish={handleLogin}>
        <Form.Item name="email" rules={[{ required: true }, { type: 'email' }]}>
          <Input prefix={<UserOutlined />} placeholder="Email" size="large" />
        </Form.Item>
        <Form.Item name="password" rules={[{ required: true }]}>
          <Input.Password prefix={<LockOutlined />} placeholder="Password" size="large" />
        </Form.Item>
        <Button type="primary" htmlType="submit" style={{ width: '100%', marginTop: 8 }} size="large" loading={loading}>
          Sign In
        </Button>
      </Form>
    </div>
  );

  const signupTab = (
    <div style={{ paddingTop: '10px' }}>
      <Form form={signupForm} layout="vertical" onFinish={handleSignup}>
        <Form.Item name="name" rules={[{ required: true, message: 'Please enter your name' }]}>
          <Input prefix={<IdcardOutlined />} placeholder="Full Name" size="large" />
        </Form.Item>
        <Form.Item name="email" rules={[{ required: true, message: 'Please enter your email' }, { type: 'email' }]}>
          <Input prefix={<MailOutlined />} placeholder="Email Address" size="large" />
        </Form.Item>
        <Form.Item name="password" rules={[{ required: true, message: 'Please set a password' }]}>
          <Input.Password prefix={<LockOutlined />} placeholder="Password" size="large" />
        </Form.Item>
        <Form.Item name="confirmPassword" rules={[{ required: true, message: 'Please confirm password' }]}>
          <Input.Password prefix={<LockOutlined />} placeholder="Confirm Password" size="large" />
        </Form.Item>
        <Button type="primary" htmlType="submit" style={{ width: '100%', marginTop: 8 }} size="large" loading={loading}>
          Create Account
        </Button>
      </Form>
    </div>
  );

  return (
    <Modal
      title={null}
      open={open}
      onCancel={onCancel}
      footer={null}
      width={400}
      destroyOnClose
    >
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        centered
        items={[
          { key: 'login', label: 'Sign In', children: loginTab },
          { key: 'signup', label: 'Create Account', children: signupTab }
        ]}
      />
    </Modal>
  );
};
