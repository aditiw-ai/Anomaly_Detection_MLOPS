import React, { createContext, useContext, useState, useEffect } from 'react';
import axios from 'axios';
import { api } from '../api/axios';

interface User {
  id: string;
  name: string;
  email: string;
  roles: string[];
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (token: string, userData: User) => void;
  logout: () => void;
  refreshUser: () => Promise<void>;
  isAuthenticated: boolean;
  hasRole: (roles: string[]) => boolean;
}

const AuthContext = createContext<AuthContextType | null>(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(localStorage.getItem('token') || localStorage.getItem('access_token'));

  useEffect(() => {
    const handleAuthExpired = () => {
      setToken(null);
      setUser(null);
      localStorage.removeItem('token');
      localStorage.removeItem('access_token');
      localStorage.removeItem('user');
    };

    window.addEventListener('shadow-hubble:auth-expired', handleAuthExpired);
    return () => window.removeEventListener('shadow-hubble:auth-expired', handleAuthExpired);
  }, []);

  const refreshUser = async () => {
    const activeToken = token || localStorage.getItem('token') || localStorage.getItem('access_token');
    if (!activeToken) return;

    try {
      const res = await api.get('/auth/me', {
        headers: { Authorization: `Bearer ${activeToken}` }
      });
      const userData: User = {
        id: res.data.id,
        name: res.data.name,
        email: res.data.email,
        roles: res.data.roles,
      };
      setUser(userData);
      localStorage.setItem('user', JSON.stringify(userData));
    } catch (e) {
      console.error("Failed to refresh user profile", e);
      if (axios.isAxiosError(e) && e.response?.status === 401) {
        logout();
      }
    }
  };

  useEffect(() => {
    if (token) {
      const storedUser = localStorage.getItem('user');
      if (storedUser) {
        try {
          setUser(JSON.parse(storedUser));
        } catch (e) {
          console.error("Failed to parse stored user", e);
        }
      }

      // Auto-refresh profile on load to get latest roles from DB
      refreshUser();

      // Setup axios interceptor for the token
      const interceptorId = api.interceptors.request.use((config) => {
        config.headers.Authorization = `Bearer ${token}`;
        return config;
      });
      return () => api.interceptors.request.eject(interceptorId);
    } else {
      setUser(null);
    }
  }, [token]);

  const login = (newToken: string, userData: User) => {
    setToken(newToken);
    setUser(userData);
    localStorage.setItem('token', newToken);
    // Keep legacy key for backward compatibility with existing sessions.
    localStorage.setItem('access_token', newToken);
    localStorage.setItem('user', JSON.stringify(userData));
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('token');
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');
  };

  const hasRole = (roles: string[]) => {
    if (!user || (!user.roles && !roles.includes('VIEWER'))) return false;
    const userRoles = user.roles || [];
    return userRoles.some(r => roles.includes(r));
  };

  return (
    <AuthContext.Provider value={{ user, token, login, logout, refreshUser, isAuthenticated: !!token, hasRole }}>
      {children}
    </AuthContext.Provider>
  );
};
