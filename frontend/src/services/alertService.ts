/**
 * Alert API Service
 * Handles all API calls for alert operations.
 */
import { api } from '../api/axios';

export interface Alert {
    id: string;
    model_id: string;
    alert_type: string;
    severity: string;
    title: string;
    message: string;
    details?: Record<string, any>;
    status: string;
    acknowledged_at?: string;
    acknowledged_by?: string;
    resolved_at?: string;
    resolution_note?: string;
    created_at: string;
}

export interface AlertsResponse {
    data: Alert[];
    meta: {
        page: number;
        page_size: number;
        total: number;
    };
    summary: {
        active: number;
        acknowledged: number;
        resolved: number;
        critical: number;
    };
}

export const alertService = {
    /**
     * List all alerts.
     */
    async listAlerts(status?: string, severity?: string): Promise<AlertsResponse> {
        const params: Record<string, string> = {};
        if (status) params.status = status;
        if (severity) params.severity = severity;

        const response = await api.get('/alerts', { params });
        return response.data;
    },

    /**
     * Get a single alert.
     */
    async getAlert(alertId: string): Promise<{ data: Alert }> {
        const response = await api.get(`/alerts/${alertId}`);
        return response.data;
    },

    /**
     * Acknowledge an alert.
     */
    async acknowledgeAlert(alertId: string, note?: string): Promise<{ data: Alert }> {
        const response = await api.post(`/alerts/${alertId}/acknowledge`, {
            resolution_note: note,
        });
        return response.data;
    },

    /**
     * Resolve an alert.
     */
    async resolveAlert(alertId: string, note?: string): Promise<{ data: Alert }> {
        const response = await api.post(`/alerts/${alertId}/resolve`, {
            resolution_note: note,
        });
        return response.data;
    },

    /**
     * Get alert statistics.
     */
    async getAlertStats(period = '7d'): Promise<any> {
        const response = await api.get(`/alerts/stats/summary?period=${period}`);
        return response.data;
    },
};
