import type { ReactNode } from 'react';
import { IconAlertTriangle, IconDatabaseOff } from '@tabler/icons-react';

/** Empty and error states: say what happened and what to do next. */
export function StateMessage({
  kind,
  title,
  children,
  action,
}: {
  kind: 'empty' | 'error';
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  const Icon = kind === 'error' ? IconAlertTriangle : IconDatabaseOff;
  return (
    <div role={kind === 'error' ? 'alert' : 'status'} className="flex flex-col items-start gap-xs py-lg">
      <Icon aria-hidden size={24} stroke={1.5} className={kind === 'error' ? 'text-down' : 'text-body'} />
      <p className="text-title-sm text-ink">{title}</p>
      {children && <div className="max-w-prose text-body-md text-body">{children}</div>}
      {action}
    </div>
  );
}
