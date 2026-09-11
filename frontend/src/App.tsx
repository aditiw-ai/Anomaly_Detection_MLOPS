/**
 * AnomaLens - Anomaly Detection MLOps Platform
*/

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfigProvider, App as AntApp } from 'antd';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { MainLayout } from './components/Layout/MainLayout';
import { ProtectedRoute } from './components/Auth/ProtectedRoute';
import { Dashboard } from './pages/Dashboard';
import Landing from './pages/Landing';
import { DataRegistry } from './pages/DataRegistry';
import { Training } from './pages/Training';
import { ModelRegistry } from './pages/ModelRegistry';
import { ModelComparison } from './pages/ModelComparison';
import { Inference } from './pages/Inference';
import { Monitoring } from './pages/Monitoring';
import { Jobs } from './pages/Jobs';
import { Retraining } from './pages/Retraining';
import { ABTesting } from './pages/ABTesting'; // Fixed named export
import { Alerts } from './pages/Alerts';
import { Settings } from './pages/Settings';

const ALL_ROLES = ['ADMIN', 'DATA_ENGINEER', 'ML_ENGINEER', 'DEPLOYER', 'VIEWER'];

const queryClient = new QueryClient({
    defaultOptions: {
        queries: { staleTime: 30000, refetchOnWindowFocus: false },
    },
});

const theme = {
    token: {
        colorPrimary: '#2563EB',
        colorSuccess: '#059669',
        colorWarning: '#D97706',
        colorError: '#DC2626',
        borderRadius: 8,
        fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
    },
};

function HomeRoute() {
    const { user } = useAuth();
    return user ? <Dashboard /> : <Landing />;
}

function App() {
    return (
        <QueryClientProvider client={queryClient}>
            <ConfigProvider theme={theme}>
                <AntApp>
                    <AuthProvider>
                        <BrowserRouter>
                            <MainLayout>
                                <Routes>
                                    {/* Home route: Landing (logged out) / Dashboard (logged in) */}
                                    <Route path="/" element={<HomeRoute />} />

                                    {/* Data — all roles */}
                                    <Route path="/data" element={
                                        <ProtectedRoute allowedRoles={ALL_ROLES}>
                                            <DataRegistry />
                                        </ProtectedRoute>
                                    } />

                                    {/* Training — all roles */}
                                    <Route path="/training" element={
                                        <ProtectedRoute allowedRoles={ALL_ROLES}>
                                            <Training />
                                        </ProtectedRoute>
                                    } />

                                    {/* Model Registry — all roles */}
                                    <Route path="/models" element={
                                        <ProtectedRoute allowedRoles={ALL_ROLES}>
                                            <ModelRegistry />
                                        </ProtectedRoute>
                                    } />

                                    {/* Compare Models — all roles */}
                                    <Route path="/models/compare" element={
                                        <ProtectedRoute allowedRoles={ALL_ROLES}>
                                            <ModelComparison />
                                        </ProtectedRoute>
                                    } />

                                    {/* Inference — all roles */}
                                    <Route path="/inference" element={
                                        <ProtectedRoute allowedRoles={ALL_ROLES}>
                                            <Inference />
                                        </ProtectedRoute>
                                    } />

                                    {/* Monitoring — all roles */}
                                    <Route path="/monitoring" element={
                                        <ProtectedRoute allowedRoles={ALL_ROLES}>
                                            <Monitoring />
                                        </ProtectedRoute>
                                    } />

                                    {/* Jobs — all roles */}
                                    <Route path="/jobs" element={
                                        <ProtectedRoute allowedRoles={ALL_ROLES}>
                                            <Jobs />
                                        </ProtectedRoute>
                                    } />

                                    {/* Retraining — all roles */}
                                    <Route path="/retraining" element={
                                        <ProtectedRoute allowedRoles={ALL_ROLES}>
                                            <Retraining />
                                        </ProtectedRoute>
                                    } />

                                    {/* A/B Testing — all roles */}
                                    <Route path="/ab-testing" element={
                                        <ProtectedRoute allowedRoles={ALL_ROLES}>
                                            <ABTesting />
                                        </ProtectedRoute>
                                    } />

                                    {/* Alerts — all roles */}
                                    <Route path="/alerts" element={
                                        <ProtectedRoute allowedRoles={ALL_ROLES}>
                                            <Alerts />
                                        </ProtectedRoute>
                                    } />

                                    {/* Admin — User Management — all roles */}
                                    <Route path="/settings" element={
                                        <ProtectedRoute allowedRoles={ALL_ROLES}>
                                            <Settings />
                                        </ProtectedRoute>
                                    } />

                                    <Route path="/admin/users" element={<Navigate to="/settings" replace />} />

                                    <Route path="*" element={<Navigate to="/" replace />} />
                                </Routes>
                            </MainLayout>
                        </BrowserRouter>
                    </AuthProvider>
                </AntApp>
            </ConfigProvider>
        </QueryClientProvider>
    );
}

export default App;
