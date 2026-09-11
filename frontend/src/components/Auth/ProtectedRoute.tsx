import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { AccessDenied } from './AccessDenied';

interface ProtectedRouteProps {
  allowedRoles: string[];
  children: React.ReactNode;
}

/**
 * Wraps a page with role-based access control.
 * Renders <AccessDenied> if the current user's role is not in allowedRoles.
 * Public pages (no login required) should NOT use this wrapper.
 */
export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ allowedRoles, children }) => {
  const { user } = useAuth();

  // Not logged in — redirect to landing/login page.
  if (!user) {
    return <Navigate to="/" replace />;
  }

  const userRoles = user.roles || [];
  const hasAccess = userRoles.some(r => allowedRoles.includes(r));

  if (!hasAccess) {
    return <AccessDenied roleName={userRoles[0]} />;
  }

  return <>{children}</>;
};
