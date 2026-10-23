import { useEffect, useState } from "react";
import { request } from "./api";
export function useRemote<T>(url: string, body?: unknown, revision = 0) {
  const serialized = body === undefined ? undefined : JSON.stringify(body);
  const key = JSON.stringify([url, serialized, revision]);
  const [state, setState] = useState<{ key: string; data?: T; error?: string }>(
    { key: "" },
  );
  useEffect(() => {
    const controller = new AbortController();
    request<T>(
      url,
      controller.signal,
      serialized === undefined ? undefined : JSON.parse(serialized),
    )
      .then((data) => {
        if (!controller.signal.aborted) setState({ key, data });
      })
      .catch((e: unknown) => {
        if (!controller.signal.aborted)
          setState({
            key,
            error: e instanceof Error ? e.message : "Request failed.",
          });
      });
    return () => controller.abort();
  }, [url, serialized, key]);
  const current = state.key === key ? state : { key };
  return { ...current, loading: !current.data && !current.error };
}
