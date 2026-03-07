import {
  Empty,
  ModelsIcon,
  Spinner,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useBudgetsQuery } from '../../hooks/useBudgetsQuery';
import type { GatewayBudget } from '../../types';

const RENEWAL_PERIOD_LABELS: Record<string, string> = {
  monthly: 'Monthly',
  quarterly: 'Quarterly',
  annually: 'Annually',
};

const formatCurrency = (amount: number, currency: string) => {
  try {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency }).format(amount);
  } catch {
    return `${currency} ${amount.toFixed(2)}`;
  }
};

const formatDate = (dateStr: string) => {
  try {
    return new Intl.DateTimeFormat('en-US', { year: 'numeric', month: 'short', day: 'numeric' }).format(
      new Date(dateStr),
    );
  } catch {
    return dateStr;
  }
};

interface BudgetsListProps {
  onBudgetClick?: (budget: GatewayBudget) => void;
}

export const BudgetsList = ({ onBudgetClick }: BudgetsListProps) => {
  const { theme } = useDesignSystemTheme();
  const { data: budgets, isLoading } = useBudgetsQuery();

  if (isLoading) {
    return (
      <div
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: theme.spacing.sm,
          padding: theme.spacing.lg,
          minHeight: 200,
        }}
      >
        <Spinner size="small" />
        <FormattedMessage defaultMessage="Loading budgets..." description="Loading message for budgets list" />
      </div>
    );
  }

  const emptyState = (
    <Empty
      image={<ModelsIcon />}
      title={<FormattedMessage defaultMessage="No budgets created" description="Empty state title for budgets list" />}
      description={
        <FormattedMessage
          defaultMessage='Use "Create budget" button to create a new budget'
          description="Empty state message for budgets list explaining how to create"
        />
      }
    />
  );

  return (
    <Table
      scrollable
      empty={budgets.length === 0 ? emptyState : undefined}
      css={{
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderRadius: theme.general.borderRadiusBase,
      }}
    >
      <TableRow isHeader>
        <TableHeader componentId="mlflow.gateway.budgets-list.name-header" css={{ flex: 2 }}>
          <FormattedMessage defaultMessage="Budget name" description="Budget name column header" />
        </TableHeader>
        <TableHeader componentId="mlflow.gateway.budgets-list.amount-header" css={{ flex: 1 }}>
          <FormattedMessage defaultMessage="Budget amount" description="Budget amount column header" />
        </TableHeader>
        <TableHeader componentId="mlflow.gateway.budgets-list.period-header" css={{ flex: 1 }}>
          <FormattedMessage defaultMessage="Renewal period" description="Renewal period column header" />
        </TableHeader>
        <TableHeader componentId="mlflow.gateway.budgets-list.period-start-header" css={{ flex: 1 }}>
          <FormattedMessage defaultMessage="Period start" description="Period start date column header" />
        </TableHeader>
        <TableHeader componentId="mlflow.gateway.budgets-list.period-end-header" css={{ flex: 1 }}>
          <FormattedMessage defaultMessage="Period end" description="Period end date column header" />
        </TableHeader>
        <TableHeader componentId="mlflow.gateway.budgets-list.spending-header" css={{ flex: 1 }}>
          <FormattedMessage defaultMessage="Spending" description="Current spending column header" />
        </TableHeader>
      </TableRow>
      {budgets.map((budget) => {
        const spendingPct = budget.amount > 0 ? (budget.current_spending / budget.amount) * 100 : 0;
        const isOverBudget = budget.current_spending > budget.amount;

        return (
          <TableRow key={budget.budget_id}>
            <TableCell css={{ flex: 2 }}>
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                <ModelsIcon css={{ color: theme.colors.textSecondary, flexShrink: 0 }} />
                {onBudgetClick ? (
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={() => onBudgetClick(budget)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        onBudgetClick(budget);
                      }
                    }}
                    css={{
                      color: theme.colors.actionPrimaryBackgroundDefault,
                      fontWeight: theme.typography.typographyBoldFontWeight,
                      cursor: 'pointer',
                      '&:hover': { textDecoration: 'underline' },
                    }}
                  >
                    {budget.name}
                  </span>
                ) : (
                  <Typography.Text bold>{budget.name}</Typography.Text>
                )}
              </div>
            </TableCell>
            <TableCell css={{ flex: 1 }}>
              <Typography.Text>{formatCurrency(budget.amount, budget.currency)}</Typography.Text>
            </TableCell>
            <TableCell css={{ flex: 1 }}>
              <Typography.Text>
                {RENEWAL_PERIOD_LABELS[budget.renewal_period] ?? budget.renewal_period}
              </Typography.Text>
            </TableCell>
            <TableCell css={{ flex: 1 }}>
              <Typography.Text>{formatDate(budget.period_start)}</Typography.Text>
            </TableCell>
            <TableCell css={{ flex: 1 }}>
              <Typography.Text>{formatDate(budget.period_end)}</Typography.Text>
            </TableCell>
            <TableCell css={{ flex: 1 }}>
              <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
                <div css={{ display: 'flex', alignItems: 'baseline', gap: theme.spacing.xs }}>
                  <Typography.Text color={isOverBudget ? 'error' : 'primary'}>
                    {formatCurrency(budget.current_spending, budget.currency)}
                  </Typography.Text>
                  <Typography.Text color="secondary" size="sm">
                    ({spendingPct.toFixed(1)}%)
                  </Typography.Text>
                </div>
                <div
                  css={{
                    height: 4,
                    borderRadius: 2,
                    backgroundColor: theme.colors.backgroundSecondary,
                    overflow: 'hidden',
                    width: '100%',
                    maxWidth: 120,
                  }}
                >
                  <div
                    css={{
                      height: '100%',
                      width: `${Math.min(spendingPct, 100)}%`,
                      backgroundColor: isOverBudget
                        ? theme.colors.actionDangerPrimaryBackgroundDefault
                        : theme.colors.actionPrimaryBackgroundDefault,
                      borderRadius: 2,
                    }}
                  />
                </div>
              </div>
            </TableCell>
          </TableRow>
        );
      })}
    </Table>
  );
};
