import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { ListBudgetsResponse } from '../types';

const queryFn = () => GatewayApi.listBudgets();

export const useBudgetsQuery = () => {
  const queryResult = useQuery<ListBudgetsResponse, Error, ListBudgetsResponse, ['gateway_budgets']>(
    ['gateway_budgets'],
    {
      queryFn,
      retry: false,
    },
  );

  return {
    data: queryResult.data?.budgets ?? [],
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    refetch: queryResult.refetch,
  };
};
