import React from 'react';
import { Result, Button } from 'antd';
import { LockOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';

interface AccessDeniedProps {
  roleName?: string;
}

export const AccessDenied: React.FC<AccessDeniedProps> = ({ roleName }) => {
  const navigate = useNavigate();
  return (
    <div style={{
      minHeight: '60vh', display: 'flex',
      alignItems: 'center', justifyContent: 'center'
    }}>
      <Result
        icon={<LockOutlined style={{ color: '#ff4d4f', fontSize: 64 }} />}
        status="403"
        title="Access Denied"
        subTitle={
          roleName
            ? `Your role (${roleName}) does not have permission to view this page.`
            : 'You do not have permission to view this page.'
        }
        extra={
          <Button type="primary" onClick={() => navigate('/')}>
            Back to Dashboard
          </Button>
        }
      />
    </div>
  );
};
